import json
import os

chrom_conversion = {
    "1": "Chr1",
    "2": "Chr2",
    "3": "Chr3",
    "4": "Chr4",
    "5": "Chr5",
    "6": "Chr6",
    "7": "Chr7",
    "8": "Chr8",
    "9": "Chr9",
    "10": "Chr10",
    "11": "Chr11",
    "12": "Chr12",
    "13": "Chr13",
    "14": "Chr14",
    "15": "Chr15",
    "16": "Chr16"
}

class Genome:
    def __init__(self, gff_path = "SGD_API/S288C/saccharomyces_cerevisiae_R64-5-1_20240529.gff"):
        self.gff_path = gff_path
        self.sequences = self.load_genome()

    def load_genome(self):
        """ Load the genome sequence (ACTG) from a gff file and save it in a dictionary
        """
        genome = {}
        with open(self.gff_path, 'r') as f:
            for line in f:
                if not line.startswith('>chr') and not line.startswith('A') and not line.startswith('C') and not line.startswith('T') and not line.startswith('G'):
                    continue
                if line.startswith('>'):
                    chrom = "C" + line[2:].strip()
                    genome[chrom] = ''
                else:
                    genome[chrom] += line.strip()
        return genome

    def get_sequence(self, chrom):
        """ Get the genome sequence for a specific chromosome
        """
        return self.sequences.get(chrom, '')
    
    def get_whole_genome(self):
        """ Get the whole genome sequence as a dictionary
        """
        return self.sequences
    
    def get_chromosome_lengths(self):
        """ Get the lengths of all chromosomes as a dictionary
        """
        lengths = {}
        for chrom in self.sequences:
            lengths[chrom] = len(self.sequences[chrom])
        return lengths

    def compute_kmer_count(self, chrom = 0, k_sizes=[1, 2, 3, 4, 5], input_file=None, output_file=None):
        """ Compute k-mer counts for a specific chromosome
        args:
            chrom: chromosome number, if 0, compute for all chromosomes
            k_sizes: list of k sizes to compute
            input_file: path to input file that contains the k-mer counts, if provided, loads counts from file
            output_file: path to output file for saving k-mer counts, if provided, saves counts to file
        returns:
            kmer_counts: dict of k sizes to dicts of k-mer counts
        """
        if input_file:
            with open(input_file, 'r') as f:
                kmer_counts = json.load(f)
            return kmer_counts
        kmer_counts = {}
        if chrom == 0:
            for chrom in self.sequences:
                sequence = self.sequences[chrom]
                for k in k_sizes:
                    if k not in kmer_counts:
                        kmer_counts[k] = {}
                    for i in range(len(sequence) - k + 1):
                        kmer = sequence[i:i + k]
                        if kmer not in kmer_counts[k]:
                            kmer_counts[k][kmer] = 0
                        kmer_counts[k][kmer] += 1
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(kmer_counts, f, indent=4)
            return kmer_counts
        else:
            chrom_name = chrom_conversion.get(str(chrom), None)
            if not chrom_name:
                print(f"Chromosome {chrom} not found.")
                return None
            sequence = self.sequences.get(chrom_name, '')
            for k in k_sizes:
                kmer_counts[k] = {}
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i:i + k]
                    if kmer not in kmer_counts[k]:
                        kmer_counts[k][kmer] = 0
                    kmer_counts[k][kmer] += 1
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(kmer_counts, f, indent=4)
            return kmer_counts
        
if __name__ == "__main__":
    genome = Genome()
    # sequences = genome.get_whole_genome()
    # for chrom in sequences:
    #     print(genome.get_chromosome_lengths()[chrom])
    genome.compute_kmer_count(chrom=0, k_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], output_file="SGD_API/architecture_info/genome_kmer_counts.json")