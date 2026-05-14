import sgd
import json
import time

class SGD_Genes:
    def __init__(self, gene_list_with_info = None, gene_list_file = None, output_file = None):
        """
        Initialize the SGD_Genes class with a dictionary of genes.

        Parameters:
        gene_list_with_info (str): JSON file containing gene information
        gene_list_file (str): Path to the file containing the list of genes.
        """
        self.genes = {}
        if gene_list_with_info:
            with open(gene_list_with_info, 'r') as f:
                self.genes = json.load(f)
        if gene_list_file:
            with open(gene_list_file, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
                for gene in genes:
                    if gene in self.genes:
                        print(f"Gene {gene} already processed, skipping.")
                        continue
                    print(f"Processing gene: {gene}")
                    try:
                        gene_name = sgd.gene(gene).details.json()["display_name"]
                    except Exception as e:
                        print(f"Error retrieving details for gene {gene}: {e}")
                        continue
                    
                    self.genes[gene] = {
                        "gene_name": gene_name,
                        "location": self.get_location(gene),
                        "essentiality": self.get_essentiality(gene),
                        "protein_domains": self.get_protein_domains(gene)
                    }
                    if output_file:
                        json.dump(self.genes, open(output_file, 'w'), indent=4)
                    else:
                        json.dump(self.genes, open(gene_list_with_info, 'w'), indent=4)
                    time.sleep(3)
                    
         
    def list_all_genes(self):
        """Return the list of all gene names."""
        return self.genes.keys()
    
    def list_all_gene_info(self):
        """Return the full dictionary of genes and their information."""
        return self.genes
    
    def list_essential_genes(self):
        """Return a list of all essential genes."""
        return [gene for gene, info in self.genes.items() if info["essentiality"]]
    
    def list_nonessential_genes(self):
        """Return a list of all non-essential genes."""
        return [gene for gene, info in self.genes.items() if not info["essentiality"]]
    
    def retrieve_gene(self, gene, display_name=False):
        """Return the full information for a given gene."""
        if display_name:
            for g, info in self.genes.items():
                if info["gene_name"] == gene:
                    gene = g
                    break
        return self.genes[gene]
    
    def retrieve_location(self, gene, display_name=False):
        """Return the location information for a given gene."""
        if display_name:
            for g, info in self.genes.items():
                if info["gene_name"] == gene:
                    gene = g
                    break
        return self.genes.get(gene, {}).get("location", None)
    
    def retrieve_essentiality(self, gene, display_name=False):
        """Return the essentiality information for a given gene."""
        if display_name:
            for g, info in self.genes.items():
                if info["gene_name"] == gene:
                    gene = g
                    break
        return self.genes.get(gene, {}).get("essentiality", None)
    
    def retrieve_protein_domains(self, gene, display_name=False):
        """Return the protein domain information for a given gene."""
        if display_name:
            for g, info in self.genes.items():
                if info["gene_name"] == gene:
                    gene = g
                    break
        return self.genes.get(gene, {}).get("protein_domains", None)
    
    def add_gene(self, gene):
        """Add a gene to the list if not already present."""
        if gene not in self.genes:
            self.genes[gene] = {
                "gene_name": gene,
                "location": self.get_location(gene),
                "essentiality": self.get_essentiality(gene),
                "protein_domains": self.get_protein_domains(gene)
            }
            
    def remove_gene(self, gene):
        """Remove a gene from the list if present."""
        if gene in self.genes:
            del self.genes[gene]

    def get_detail(self, gene, endpoint):
        """
        Generic accessor for any SGD REST endpoint for a given gene.
        
        Parameters:
        gene (str): Gene name (systematic or standard name).
        endpoint (str): One of sgd.gene.endpoints (e.g. "details", "phenotype_details").
        
        Returns:
        dict: JSON response from the API.
        """
        g = sgd.gene(gene)
        if endpoint not in g.endpoints:
            raise ValueError(f"Unknown endpoint '{endpoint}'. Available: {list(g.endpoints)}")
        return getattr(g, endpoint).json()

    def get_location(self, gene):
        """
        Return chromosome, start, end, and strand for a given gene
        (from the S288C reference genome).
        """
        details = sgd.gene(gene).sequence_details.json()
        chrom = details["genomic_dna"][0]["contig"]["format_name"]
        start = details["genomic_dna"][0]["start"]
        end = details["genomic_dna"][0]["end"]
        return {
            "chromosome": chrom,
            "start": start,
            "end": end
        }

    def get_essentiality(self, gene, strain = "S288C"):
        """
        Retrieve essentiality information based on phenotype annotations.
        Returns True if there is an inviability/null phenotype.
        """
        phenotypes = sgd.gene(gene).phenotype_details.json()
        for p in phenotypes:
            if p["strain"]["display_name"] == strain:
                if p["mutant_type"] == "null" and p["phenotype"]["display_name"] == "inviable":
                    return True
                elif p["mutant_type"] == "null" and p["phenotype"]["display_name"] == "viable":
                    return False
            else:
                continue
        return False

    def get_protein_domains(self, gene):
        """
        Retrieve protein domain information for a given gene.
        """
        domains_json = sgd.gene(gene).protein_domain_details.json()
        domains = {}
        for d in domains_json:
            if d["domain"]["display_name"] in domains:
                domains[d["domain"]["display_name"]]["description"].append(d["domain"]["description"])
                domains[d["domain"]["display_name"]]["start"].append(d["start"])
                domains[d["domain"]["display_name"]]["end"].append(d["end"])
            domains[d["domain"]["display_name"]] = {}
            domains[d["domain"]["display_name"]]["description"] = [d["domain"]["description"]]
            domains[d["domain"]["display_name"]]["start"] = [d["start"]]
            domains[d["domain"]["display_name"]]["end"] = [d["end"]]
        return domains



# sgd_genes = SGD_Genes(gene_list_with_info="SGD_API/architecture_info/yeast_genes_with_info.json")
# print(sgd_genes.retrieve_gene("YAL001C"))
# print(sgd_genes.retrieve_location("YAL001C"))
# print(sgd_genes.retrieve_essentiality("YAL001C"))
# print(sgd_genes.retrieve_protein_domains("YAL001C"))
# print(sgd_genes.retrieve_gene("TFC3", display_name=True))
# print(sgd_genes.retrieve_location("TFC3", display_name=True))
# print(sgd_genes.retrieve_essentiality("TFC3", display_name=True))
# print(sgd_genes.retrieve_protein_domains("TFC3", display_name=True))
# assert sgd_genes.retrieve_gene("YAL001C") == sgd_genes.retrieve_gene("TFC3", display_name=True)
# assert sgd_genes.retrieve_location("YAL001C") == sgd_genes.retrieve_location("TFC3", display_name=True)
# assert sgd_genes.retrieve_essentiality("YAL001C") == sgd_genes.retrieve_essentiality("TFC3", display_name=True)
# assert sgd_genes.retrieve_protein_domains("YAL001C") == sgd_genes.retrieve_protein_domains("TFC3", display_name=True)



