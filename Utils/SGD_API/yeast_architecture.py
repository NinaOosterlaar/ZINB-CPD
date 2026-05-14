import sgd
import json
import os
from collections import defaultdict
import numpy as np

representative_genes = [
    "PRM9",  # Chr I
    "ALG3",   # Chr II
    "CDC10",     # Chr III
    "HO",   # Chr IV
    "RNR1",   # Chr V
    "SMC1",   # Chr VI
    "CUP2",   # Chr VII
    "OPI1",   # Chr VIII
    "ATG32",   # Chr IX
    "TDH2",   # Chr X
    "MLP1",   # Chr XI
    "RDN25-1",# Chr XII
    "PHO84",  # Chr XIII
    "MRPL50",  # Chr XIV
    "HIS3",   # Chr XV
    "SSN3"   # Chr XVI
]

mapping_to_roman = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    13: "XIII",
    14: "XIV",
    15: "XV",
    16: "XVI"
}

chromosome_length = {
    "ChrI": 230218,
    "ChrII": 813184,
    "ChrIII": 316620,
    "ChrIV": 1531933,
    "ChrV": 576874,
    "ChrVI": 270161,
    "ChrVII": 1090940,
    "ChrVIII": 562643,
    "ChrIX": 439888,
    "ChrX": 745751,
    "ChrXI": 666816,
    "ChrXII": 1078171,
    "ChrXIII": 924431,
    "ChrXIV": 784333,
    "ChrXV": 1091291,
    "ChrXVI": 948066,
    "ChrM": 85779,          # mitochondrial genome (approx for S288C)
    "2micron": 6318         # 2-micron plasmid
}

def create_centromere_dict(output_file):
    """Create a dictionary of centromere locations for each chromosome."""
    centromeres = {}
    for i, gene in enumerate(representative_genes, start=1):
        chrom_name = "Chromosome_" + mapping_to_roman[i]
        info = sgd.gene(gene).sequence_details.json()
        start = info['genomic_dna'][0]["contig"]['centromere_start']
        end = info['genomic_dna'][0]["contig"]['centromere_end']
        middle = (start + end) // 2
        length = end - start
        centromeres[chrom_name] = {
            "start": start,
            "end": end,
            "middle": middle,
            "length": length
        }
    with open(output_file, 'w') as f:
        json.dump(centromeres, f, indent=4)
        
def create_nucleosome_dict(input_file, output_dir):
    """Create a dictionary of nucleosome positions for each chromosome.
    A separate file is created for each chromosome in the specified directory.
    """
    nucleosomes = {}
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # Skip header lines
            parts = line.strip().split('\t')
            if parts[0] not in nucleosomes:
                nucleosomes[parts[0]] = []
            start = int(parts[3])
            end = int(parts[4])
            middle = (start + end) // 2
            fuzzy = True if "fuzzy" in parts[8] else False
            nucleosomes[parts[0]].append((start, middle, end, fuzzy))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for chrom in nucleosomes:
        print(chrom)
        number = chrom.replace("chr", "")
        chrom_name = mapping_to_roman[int(number)]
        file_path = os.path.join(output_dir, f"Chr{chrom_name}.json")
        with open(file_path, 'w') as f:
            json.dump(nucleosomes[chrom], f, indent=4)


def create_nucleosome_dict_nature(input_file, output_dir, nucleosome_width=147):
    """Create a dictionary of nucleosome positions for each chromosome from Nature file.
    
    The Nature file format (2013 data) contains:
    - chromosome name (e.g., 'chrI')
    - nucleosome center position
    - occupancy value
    
    A separate file is created for each chromosome in the specified directory.
    The nucleosome_width parameter defines the typical nucleosome size (default 147 bp).
    """
    nucleosomes = {}
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            chrom = parts[0]
            if chrom not in nucleosomes:
                nucleosomes[chrom] = []
            
            middle = int(parts[1])
            
            # Calculate start and end positions based on nucleosome width
            # Assuming the position is the center
            half_width = nucleosome_width // 2
            start = middle - half_width
            end = middle + half_width
            
            # All nucleosomes from this dataset are considered not fuzzy
            fuzzy = False
            
            nucleosomes[chrom].append((start, middle, end, fuzzy))
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save nucleosome positions for each chromosome
    for chrom in nucleosomes:
        print(chrom)
        # Convert chromosome name from 'chrI', 'chrII', etc. to Roman numerals
        number = chrom.replace("chr", "")
        if number.upper() in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]:
            chrom_name = number.upper()
        else:
            # Handle numeric format (e.g., 'chr1' -> 'I')
            try:
                chrom_name = mapping_to_roman[int(number)]
            except (ValueError, KeyError):
                print(f"Warning: Could not convert chromosome name: {chrom}")
                continue
        
        # Save nucleosome positions (same format as 2009 data)
        file_path = os.path.join(output_dir, f"Chr{chrom_name}.json")
        with open(file_path, 'w') as f:
            json.dump(nucleosomes[chrom], f, indent=4)
        

        
class Centromeres:
    def __init__(self, centromere_file = "Utils/SGD_API/architecture_info/centromeres.json"):
        with open(centromere_file, 'r') as f:
            self.centromeres = json.load(f)
    
    def get_centromere(self, chromosome):
        """Return the centromere information for a given chromosome."""
        return self.centromeres.get(chromosome, None)
    
    def get_middle(self, chromosome):
        """Return the middle position of the centromere for a given chromosome."""
        centromere = self.get_centromere(chromosome)
        return centromere["middle"] if centromere else None
    
    def get_length(self, chromosome):
        """Return the length of the centromere for a given chromosome."""
        centromere = self.get_centromere(chromosome)
        return centromere["length"] if centromere else None
    
    def get_start(self, chromosome):
        """Return the start position of the centromere for a given chromosome."""
        centromere = self.get_centromere(chromosome)
        return centromere["start"] if centromere else None
    
    def get_end(self, chromosome):
        """Return the end position of the centromere for a given chromosome."""
        centromere = self.get_centromere(chromosome)
        return centromere["end"] if centromere else None
    
    def compute_distance(self, chrom, position):
        """Compute the distance from a given position to the centromere middle on the specified chromosome."""
        centromere = self.get_centromere(chrom)
        if centromere:
            return position - centromere["middle"]
        return None

    def list_all_centromeres(self):
        """Return the full dictionary of centromeres."""
        return self.centromeres

    def retrieve_all_middles(self):
        """Return a list of all centromere middles."""
        middles = {}
        for chrom in self.centromeres:
            middles[chrom] = self.centromeres[chrom]["middle"]
        return middles

    def retrieve_all_lengths(self):
        """Return a list of all centromere lengths."""
        lengths = {}
        for chrom in self.centromeres:
            lengths[chrom] = self.centromeres[chrom]["length"]
        return lengths
    
    def retrieve_all_starts(self):
        """Return a list of all centromere starts."""
        starts = {}
        for chrom in self.centromeres:
            starts[chrom] = self.centromeres[chrom]["start"]
        return starts
    
    def retrieve_all_ends(self):
        """Return a list of all centromere ends."""
        ends = {}
        for chrom in self.centromeres:
            ends[chrom] = self.centromeres[chrom]["end"]
        return ends
        
    
class Nucleosomes:
    def __init__(self, nucleosome_dir="Utils/SGD_API/nucleosome_data/2013/"):
        """Load all chromosome nucleosome files from a directory."""
        self.nucleosomes = {}
        print(nucleosome_dir)
        for chrom in chromosome_length.keys():
            file_path = os.path.join(nucleosome_dir, f"{chrom}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.nucleosomes[chrom] = json.load(f)
        print(f"Loaded nucleosome data for chromosomes: {list(self.nucleosomes.keys())}")
    
    def get_nucleosomes(self, chrom):
        """Return the list of nucleosomes for a given chromosome."""
        return self.nucleosomes.get(chrom, [])
    
    def list_all_nucleosomes(self):
        """Return all nucleosomes for all chromosomes."""
        return self.nucleosomes
    
    def get_all_middles(self):
        """Return dict of chromosome names → list of nucleosome middles."""
        return {chrom: [nuc[1] for nuc in nucs] for chrom, nucs in self.nucleosomes.items()}
    
    def count_nucleosomes(self, chrom):
        """Return the number of nucleosomes on a given chromosome."""
        return len(self.nucleosomes.get(chrom, []))
    
    def get_middles(self, chrom):
        """Return a list of nucleosome middles on a given chromosome."""
        return [nuc[1] for nuc in self.nucleosomes.get(chrom, [])]
    
    def compute_average_span(self, chrom):
        """Compute the average span of nucleosomes on a given chromosome."""
        number_of_nucleosomes = self.count_nucleosomes(chrom)
        return chromosome_length[chrom] / number_of_nucleosomes if number_of_nucleosomes > 0 else 0
    
    def compute_distance(self, chrom, position):
        """Compute the distance to the nearest nucleosome on a given chromosome from a specified position.
        The nucleosomes are in order of their start positions, however fuzzy nucleosomes are included after the well-positioned ones.
        
        Args:
            chrom (str): Chromosome name (e.g., 'ChrI', 'ChrII')
            position (int): Position to compute distance from
            
        Returns:
            int: Minimum distance to nearest nucleosome center, or None if no nucleosomes found
        """
        
        nucleosomes = self.get_nucleosomes(chrom)
        if not nucleosomes: return None
        
        # Extract all middle positions - this handles both fuzzy and non-fuzzy nucleosomes
        middles = np.array([nuc[1] for nuc in nucleosomes])
        
        # Use vectorized NumPy operations for efficiency with large datasets
        distances = np.abs(middles - position)
        min_distance = np.min(distances)
        
        return int(min_distance)
    
    
    def compute_exposure(self, chrom, folder = "Utils/SGD_API/nucleosome_data/2013/"):
        """ Compute how often each distance from a nucleosome occurs on a chromosome
        
        Args:
            chrom (str): Chromosome name (e.g., 'ChrI', 'ChrII')
        Returns:
            dict: Dictionary with distances as keys and their exposure frequency as values
        """
        exposure_file = f"{folder}{chrom}_exposure.json"
        if os.path.exists(exposure_file):
            temp = json.load(open(exposure_file, 'r'))
            # Make sure keys are integers
            temp = {int(k): v for k, v in temp.items()}
            return temp
        else:
            nucleosome_count = self.count_nucleosomes(chrom)
            if nucleosome_count == 0: return {}
            distance_counts = {}
            for position in range(1, chromosome_length[chrom] + 1):
                distance = self.compute_distance(chrom, position)
                if distance in distance_counts:
                    distance_counts[distance] += 1
                else:
                    distance_counts[distance] = 1
            json.dump(distance_counts, open(f"{folder}{chrom}_exposure.json", 'w'), indent=4)
            return distance_counts


if __name__ == "__main__":
    # Example usage: Create 2013 nucleosome data from Nature file
    # nature_file = "Utils/SGD_API/nucleosome_data/41586_2012_BFnature11142_MOESM263_ESM.txt"
    # output_dir_2013 = "Utils/SGD_API/nucleosome_data/2013"
    
    # # Create the 2013 nucleosome dictionary
    # create_nucleosome_dict_nature(nature_file, output_dir_2013)
    # print(f"\n2013 nucleosome data created successfully in {output_dir_2013}")
    nucleosomes = Nucleosomes()
    for chrom in chromosome_length.keys():
        exposure = nucleosomes.compute_exposure(chrom)
        print(f"Computed exposure for {chrom}, number of unique distances: {len(exposure)}")
