import sys
import os
import json
import statistics
from pathlib import Path

# Fix PYTHONPATH so we can import from src
sys.path.append(os.getcwd())

def analyze_all_caches(cache_dir: str, min_chars: int = 500, max_chars: int = 1500):
    cache_path = Path(cache_dir)
    atom_files = list(cache_path.glob("*_atoms.json"))
    
    if not atom_files:
        print(f"No *_atoms.json files found in {cache_dir}")
        return

    all_atom_lengths = []
    total_raw_atoms = 0
    total_packed_chunks = 0
    all_packed_lengths = []
    
    print(f"\nProcessing {len(atom_files)} cache files...")
    
    from src.common.parsing.packer import greedy_pack
    
    for f_path in atom_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            atoms = [a["text"] for a in data.get("atoms_data", [])]
            if not atoms:
                continue
                
            all_atom_lengths.extend([len(a) for a in atoms])
            total_raw_atoms += len(atoms)
            
            # Simulate packing for this file
            packed = greedy_pack(atoms, min_chars=min_chars, max_chars=max_chars)
            total_packed_chunks += len(packed)
            all_packed_lengths.extend([len(p) for p in packed])
            
        except Exception as e:
            print(f"Error processing {f_path.name}: {e}")

    if not all_atom_lengths:
        print("No atom data found.")
        return

    # Aggregate Statistics
    necessity_score = (len([l for l in all_atom_lengths if l < min_chars]) / total_raw_atoms) * 100
    
    print(f"\n{'='*60}")
    print(f"AGGREGATED PACKER ANALYSIS ({len(atom_files)} Files)")
    print(f"{'='*60}")
    print(f"Total Raw Atoms (Total Parsed Units): {total_raw_atoms}")
    print(f"Global Necessity Score: {necessity_score:.1f}% of atoms are below {min_chars} chars")
    print(f"Conclusion: {'CRITICAL' if necessity_score > 50 else 'MODERATE'} need for packing.")
    
    print(f"\nRaw Statistics (Before Packing):")
    print(f"  Mean Length:   {statistics.mean(all_atom_lengths):.1f} chars")
    print(f"  Median Length: {statistics.median(all_atom_lengths)} chars")
    
    print(f"\nEfficiency after Packing:")
    print(f"  Total Final Chunks:  {total_packed_chunks}")
    print(f"  Storage Reduction:    {((total_raw_atoms - total_packed_chunks) / total_raw_atoms) * 100:.1f}%")
    print(f"  Final Mean Length:    {statistics.mean(all_packed_lengths):.1f} chars")
    print(f"  Final Median Length:  {statistics.median(all_packed_lengths)} chars")

if __name__ == "__main__":
    cache_dir = "/Users/sercan/Projects/RAG-poc/data_lake/parse_cache"
    analyze_all_caches(cache_dir)
