#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
import tempfile
import glob
from pathlib import Path
import subprocess
import torch
import numpy as np
import json

def create_temp_structure(temp_dir):
    """Create the required directory structure in the temporary directory"""
    os.makedirs(os.path.join(temp_dir, "pdb"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "embedding", "temp"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "embedding", "pH"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "embedding", "salt"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "DSSP"), exist_ok=True)
    return temp_dir

def link_or_copy_files(source_dir, target_dir, file_pattern="*", symlink=True, count=False):
    """Link or copy files from source directory to target directory"""
    if not source_dir:
        return 0 if count else None
        
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    if not source_dir.exists():
        print(f"Warning: Source directory {source_dir} does not exist. Skipping.")
        return 0 if count else None
    
    file_count = 0
    for file_path in source_dir.glob(file_pattern):
        target_path = target_dir / file_path.name
        
        if target_path.exists():
            continue
            
        if symlink:
            try:
                os.symlink(file_path.absolute(), target_path)
                file_count += 1
            except OSError:
                # Fall back to copying if symlinks are not supported
                shutil.copy2(file_path, target_path)
                file_count += 1
        else:
            shutil.copy2(file_path, target_path)
            file_count += 1
    
    return file_count if count else None

def extract_xyz_from_pdb_multi_atom(pdb_file):
    """
    Extract XYZ coordinates from PDB file with the correct format for GeoPoc.
    The tensor format should be [N, 5, 3] where:
    - N is the number of residues
    - 5 is the number of atoms per residue (N, CA, C, O, CB)
    - 3 is the XYZ coordinates
    
    If some atoms are missing, we'll use dummy coordinates (zeros).
    """
    try:
        with open(pdb_file, 'r') as f:
            pdb_content = f.readlines()
        
        # Group atoms by residue
        residues = {}
        for line in pdb_content:
            if not line.startswith("ATOM"):
                continue
                
            residue_id = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            
            # Only extract backbone atoms and CB
            if atom_name not in ["N", "CA", "C", "O", "CB"]:
                continue
                
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            if residue_id not in residues:
                residues[residue_id] = {}
                
            residues[residue_id][atom_name] = [x, y, z]
        
        # Create ordered list of residues
        ordered_residues = sorted(residues.keys())
        
        # Create ordered 3D tensor with shape [n_residues, 5, 3]
        n_residues = len(ordered_residues)
        xyz_tensor = np.zeros((n_residues, 5, 3), dtype=np.float32)
        
        # Fill in coordinates for each residue and atom
        atom_indices = {"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4}
        
        for i, res_id in enumerate(ordered_residues):
            for atom_name, coords in residues[res_id].items():
                atom_idx = atom_indices.get(atom_name)
                if atom_idx is not None:
                    xyz_tensor[i, atom_idx, :] = coords
        
        print(f"Extracted coordinates for {n_residues} residues with shape {xyz_tensor.shape}")
        return xyz_tensor
    
    except Exception as e:
        print(f"Error processing PDB file {pdb_file}: {e}")
        raise e

def preprocess_pdb_to_tensor(pdb_dir, validate_tensors=True):
    """Convert PDB files to tensor files to avoid ESMFold loading"""
    count = 0
    validation_success = True
    pdb_files = list(Path(pdb_dir).glob("*.pdb"))
    
    if not pdb_files:
        print("No PDB files found for tensor conversion.")
        return 0, False
    
    print(f"Converting {len(pdb_files)} PDB files to tensor format...")
    
    # Process each PDB file
    for pdb_file in pdb_files:
        tensor_file = pdb_file.with_suffix('.tensor')
        
        # Skip if tensor already exists
        if tensor_file.exists():
            continue
        
        # Convert PDB to tensor
        try:
            # Use the new function to extract coordinates with correct shape
            xyz = extract_xyz_from_pdb_multi_atom(pdb_file)
            
            # Save as tensor
            tensor = torch.tensor(xyz, dtype=torch.float32)
            torch.save(tensor, tensor_file)
            print(f"Saved tensor with shape {tensor.shape} for {pdb_file.name}")
            count += 1
            
        except Exception as e:
            print(f"Error converting {pdb_file.name} to tensor: {e}")
    
    print(f"Successfully converted {count} PDB files to tensor format.")
    return count, validation_success

def inspect_embedding_file(file_path):
    """Inspect an embedding file to determine its structure"""
    try:
        data = torch.load(file_path)
        print(f"\nEmbedding file structure for {os.path.basename(file_path)}:")
        
        if isinstance(data, dict):
            print("- Top level is a dictionary with keys:", list(data.keys()))
            
            if 'representations' in data:
                print("- 'representations' is a dictionary with keys:", list(data['representations'].keys()))
                
                # Get the last layer if no layer 36
                available_layers = list(data['representations'].keys())
                if isinstance(available_layers[0], int):
                    print(f"- Available layers: {available_layers}")
                    last_layer = max(available_layers)
                    print(f"- Last layer: {last_layer}")
                    
                    if 36 not in available_layers:
                        print(f"- WARNING: Layer 36 not found! GeoPoc expects layer 36.")
                        return last_layer
        else:
            print("- Not a dictionary! Type:", type(data))
            
        return 36  # Default to layer 36 if everything seems normal
    except Exception as e:
        print(f"Error inspecting embedding file: {e}")
        return 36  # Default to layer 36

def fix_embedding_files(embedding_dir, task, output_dir, protein_ids=None):
    """Preprocess embedding files to make them compatible with GeoPoc"""
    if not os.path.exists(embedding_dir):
        print("Embedding directory does not exist!")
        return False
        
    # Check if there are embedding files
    embedding_files = list(Path(embedding_dir).glob("*.pt"))
    if not embedding_files:
        print("No embedding files found!")
        return False
        
    print(f"Found {len(embedding_files)} embedding files. Inspecting...")
    
    # Inspect first embedding file to determine structure
    sample_file = embedding_files[0]
    layer_to_use = inspect_embedding_file(sample_file)
    
    if layer_to_use != 36:
        print(f"\nWARNING: Using layer {layer_to_use} instead of 36 which GeoPoc expects.")
        print("This is an automated fix attempt.")
    
    # Create task-specific output directory if it doesn't exist
    task_dir = os.path.join(output_dir, task)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)
    
    # Load ESM min-max values for normalization
    try:
        # This path should match where GeoPoc expects to find the file
        esm_min_max_path = "/app/GeoPoc/feature_extraction/ESM_Min_Max.pkl"
        import pickle
        ESM_MIN_MAX = pickle.load(open(esm_min_max_path, 'rb'))
        MIN = ESM_MIN_MAX[f"{task}_Min"]
        MAX = ESM_MIN_MAX[f"{task}_Max"]
        print(f"Loaded normalization parameters for task '{task}'")
    except Exception as e:
        print(f"Error loading normalization values: {e}")
        print("Cannot normalize embeddings. GeoPoc may fail later.")
        return False
    
    # Process each embedding file
    success_count = 0
    for emb_file in embedding_files:
        protein_id = os.path.basename(emb_file).replace('.pt', '')
        
        # Skip if not in protein_ids list (when provided)
        if protein_ids and protein_id not in protein_ids:
            continue
            
        try:
            # Load the embedding
            raw_esm = torch.load(emb_file)
            
            # Handle different embedding formats
            if isinstance(raw_esm, dict) and 'representations' in raw_esm:
                try:
                    # Try to get the specified layer
                    emb_data = raw_esm['representations'][layer_to_use].numpy()
                except KeyError:
                    # If specific layer fails, try the first available layer
                    available_layers = list(raw_esm['representations'].keys())
                    if available_layers:
                        emb_data = raw_esm['representations'][available_layers[0]].numpy()
                    else:
                        raise ValueError("No representation layers found")
            else:
                # If not in expected format, try to use the data as is
                emb_data = np.array(raw_esm)
            
            # Normalize the embedding
            esm_emb = (emb_data - MIN) / (MAX - MIN)
            
            # Save normalized embedding
            output_file = os.path.join(task_dir, f"{protein_id}.tensor")
            torch.save(torch.tensor(esm_emb, dtype=torch.float32), output_file)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing embedding for {protein_id}: {e}")
    
    print(f"Successfully processed {success_count} out of {len(embedding_files)} embedding files")
    return success_count > 0

def main():
    parser = argparse.ArgumentParser(description="GeoPoc prediction with flexible feature paths")
    
    # Original GeoPoc arguments
    parser.add_argument("-i", "--dataset_path", type=str, required=True, 
                        help="Path to the input FASTA file")
    parser.add_argument("-o", "--output_path", type=str, default="./output/",
                        help="Path to store the prediction results")
    parser.add_argument("--task", type=str, default="temp", choices=["temp", "pH", "salt"],
                        help="Prediction task: temperature, pH, or salt")
    parser.add_argument("--model_path", type=str, default="/app/GeoPoc/model/",
                        help="Path to model weights")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID to use")
    
    # New arguments for feature paths (all optional)
    parser.add_argument("--pdb_dir", type=str, default=None,
                        help="Optional: Directory containing PDB files (.pdb or .tensor)")
    parser.add_argument("--embedding_dir", type=str, default=None,
                        help="Optional: Directory containing ESM embedding files (.pt)")
    parser.add_argument("--norm_embedding_dir", type=str, default=None,
                        help="Optional: Directory containing task-specific normalized embeddings (.tensor)")
    parser.add_argument("--dssp_dir", type=str, default=None,
                        help="Optional: Directory containing DSSP feature files (.tensor)")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary files after prediction")
    parser.add_argument("--copy_files", action="store_true",
                        help="Copy files instead of creating symlinks")
    parser.add_argument("--skip_tensor_conversion", action="store_true",
                        help="Skip automatic PDB to tensor conversion")
    parser.add_argument("--force_esmfold", action="store_true",
                        help="Force using ESMFold even if PDB files exist")
    parser.add_argument("--fix_embeddings", action="store_true",
                        help="Attempt to fix embedding files to match GeoPoc's expected format")
    
    args = parser.parse_args()
    
    # Read the FASTA file to get protein IDs
    try:
        print(f"Reading FASTA file: {args.dataset_path}")
        with open(args.dataset_path, 'r') as f:
            protein_ids = []
            for line in f:
                if line.startswith('>'):
                    protein_ids.append(line.strip()[1:])
        print(f"Found {len(protein_ids)} protein sequences")
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        protein_ids = []
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create temporary directory with required structure
    temp_dir = tempfile.mkdtemp(prefix="geopoc_features_")
    print(f"Created temporary directory: {temp_dir}")
    feature_dir = create_temp_structure(temp_dir)
    
    # Set up feature paths (same as in original GeoPoc code)
    pdb_path = os.path.join(feature_dir, "pdb")
    embedding_path = os.path.join(feature_dir, "embedding")
    dssp_path = os.path.join(feature_dir, "DSSP")
    norm_emb_path = os.path.join(embedding_path, args.task)
    
    # Track what features will need to be computed
    missing_features = {
        'pdb': len(protein_ids),
        'embedding': len(protein_ids),
        'norm_embedding': len(protein_ids),
        'dssp': len(protein_ids)
    }
    
    # Link or copy existing PDB files
    if args.pdb_dir:
        print(f"Using PDB files from: {args.pdb_dir}")
        # Copy both .pdb and .tensor files
        linked_pdbs = link_or_copy_files(args.pdb_dir, pdb_path, "*.pdb", not args.copy_files, count=True)
        linked_tensors = link_or_copy_files(args.pdb_dir, pdb_path, "*.tensor", not args.copy_files, count=True)
        print(f"  - Linked/copied {linked_pdbs} PDB files and {linked_tensors} tensor files")
        
        # NEW: Preprocess PDB files to tensors to avoid ESMFold loading
        if not args.skip_tensor_conversion and linked_pdbs > 0:
            processed_count, validation_success = preprocess_pdb_to_tensor(pdb_path)
            print(f"Preprocessed {processed_count} PDB files to tensors")
            
            if not validation_success:
                print("NOTE: Tensor validation couldn't be performed or failed.")
                print("If GeoPoc still loads ESMFold, use --skip_tensor_conversion flag")
        
        # Count how many proteins still need PDB generation
        if protein_ids:
            pdb_missing = 0
            for protein_id in protein_ids:
                tensor_exists = os.path.exists(os.path.join(pdb_path, f"{protein_id}.tensor"))
                pdb_exists = os.path.exists(os.path.join(pdb_path, f"{protein_id}.pdb"))
                if not tensor_exists and not pdb_exists:
                    pdb_missing += 1
            missing_features['pdb'] = pdb_missing
            print(f"  - {pdb_missing} out of {len(protein_ids)} proteins will need structure prediction")
    else:
        print("No PDB directory provided - structures will be predicted by ESMFold")
    
    # Link or copy existing embedding files
    if args.embedding_dir:
        print(f"Using ESM embeddings from: {args.embedding_dir}")
        linked_embs = link_or_copy_files(args.embedding_dir, embedding_path, "*.pt", not args.copy_files, count=True)
        print(f"  - Linked/copied {linked_embs} embedding files")
        
        # NEW: Fix embedding files if requested
        if args.fix_embeddings:
            print("\nAttempting to fix and normalize embedding files...")
            fix_success = fix_embedding_files(
                embedding_path, 
                args.task, 
                embedding_path, 
                protein_ids
            )
            
            if fix_success:
                print("Successfully pre-normalized embeddings for GeoPoc")
            else:
                print("WARNING: Failed to pre-normalize embeddings")
        
        # Count how many proteins still need embedding generation
        if protein_ids:
            emb_missing = 0
            for protein_id in protein_ids:
                if not os.path.exists(os.path.join(embedding_path, f"{protein_id}.pt")):
                    emb_missing += 1
            missing_features['embedding'] = emb_missing
            print(f"  - {emb_missing} out of {len(protein_ids)} proteins will need embedding generation")
    else:
        print("No embedding directory provided - embeddings will be generated by ESM")
    
    # Link or copy existing normalized embedding files
    if args.norm_embedding_dir:
        print(f"Using normalized {args.task} embeddings from: {args.norm_embedding_dir}")
        linked_norm_embs = link_or_copy_files(args.norm_embedding_dir, norm_emb_path, "*.tensor", not args.copy_files, count=True)
        print(f"  - Linked/copied {linked_norm_embs} normalized embedding files")
        
        # Count how many proteins still need normalized embedding generation
        if protein_ids:
            norm_emb_missing = 0
            for protein_id in protein_ids:
                if not os.path.exists(os.path.join(norm_emb_path, f"{protein_id}.tensor")):
                    norm_emb_missing += 1
            missing_features['norm_embedding'] = norm_emb_missing
            print(f"  - {norm_emb_missing} out of {len(protein_ids)} proteins will need normalized embedding generation")
    else:
        print(f"No normalized embedding directory provided - {args.task}-specific embeddings will be generated")
    
    # Link or copy existing DSSP files
    if args.dssp_dir:
        print(f"Using DSSP features from: {args.dssp_dir}")
        linked_dssp = link_or_copy_files(args.dssp_dir, dssp_path, "*.tensor", not args.copy_files, count=True)
        print(f"  - Linked/copied {linked_dssp} DSSP feature files")
        
        # Count how many proteins still need DSSP generation
        if protein_ids:
            dssp_missing = 0
            for protein_id in protein_ids:
                if not os.path.exists(os.path.join(dssp_path, f"{protein_id}.tensor")):
                    dssp_missing += 1
            missing_features['dssp'] = dssp_missing
            print(f"  - {dssp_missing} out of {len(protein_ids)} proteins will need DSSP feature generation")
    else:
        print("No DSSP directory provided - secondary structure features will be calculated")
    
    # Build the predict.py command
    cmd = [
        "python", "/app/GeoPoc/predict.py",
        "-i", args.dataset_path,
        "--feature_path", f"{feature_dir}/",
        "--model_path", args.model_path,
        "-o", args.output_path,
        "--task", args.task,
        "--gpu", args.gpu
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run the prediction
    try:
        subprocess.run(cmd, check=True)
        print(f"Prediction completed successfully. Results saved to {args.output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary directory
        if not args.keep_temp:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        else:
            print(f"Temporary directory kept at: {temp_dir}")

if __name__ == "__main__":
    main()