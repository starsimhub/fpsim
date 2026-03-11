import numpy as np


class ContraceptiveChoice:
    def __init__(self):
        # Initialize with base methods: no_method, condoms, pill
        self.methods = ['no_method', 'condoms', 'pill']
        self.method_index = {m: i for i, m in enumerate(self.methods)}
        
        # Example switching matrix (rows = from, cols = to)
        # Each row sums to 1
        self.switching_matrix = np.array([
            [0.7, 0.2, 0.1],  # from no_method
            [0.1, 0.7, 0.2],  # from condoms
            [0.05, 0.15, 0.8] # from pill
        ])
        
    def add_method(self, year, method, copy_from_row, copy_from_col, 
                   initial_share=0.0, renormalize=True):
        """
        Add a new contraceptive method and expand the switching matrix.
        """
        if method in self.methods:
            raise ValueError(f"Method '{method}' already exists in the simulation")
        
        if copy_from_row not in self.methods:
            raise ValueError(f"copy_from_row '{copy_from_row}' not found")
        
        if copy_from_col not in self.methods:
            raise ValueError(f"copy_from_col '{copy_from_col}' not found")
        
        row_idx = self.method_index[copy_from_row]
        col_idx = self.method_index[copy_from_col]
        
        n = len(self.methods)
        new_matrix = np.zeros((n + 1, n + 1))

        # Copy existing matrix to upper-left block
        new_matrix[:n, :n] = self.switching_matrix
        
        # Add new row (from new method to all methods)
        # Copy probabilities from copy_from_row
        new_matrix[n, :n] = self.switching_matrix[row_idx, :]
        new_matrix[n, n] = initial_share  # probability of staying with new method
        
        # Add new column (from all methods to new method)
        # Copy probabilities from copy_from_col
        new_matrix[:n, n] = self.switching_matrix[:, col_idx]
        
        if renormalize:
            # Renormalize each row to sum to 1
            row_sums = new_matrix.sum(axis=1, keepdims=True)
            new_matrix = new_matrix / row_sums
        
        # Update the object state
        self.switching_matrix = new_matrix
        self.methods.append(method)
        self.method_index[method] = n
        
        print(f"Added method '{method}' in year {year}")
        print(f"New switching matrix shape: {self.switching_matrix.shape}")
        print(f"Available methods: {self.methods}")
        
    def display_matrix(self):
        """Display the switching matrix with labeled rows and columns."""
        print("\nSwitching Matrix (rows=from, cols=to):")
        print(f"{'':15s}", end='')
        for m in self.methods:
            print(f"{m:12s}", end='')
        print()
        
        for i, from_method in enumerate(self.methods):
            print(f"{from_method:15s}", end='')
            for j in range(len(self.methods)):
                print(f"{self.switching_matrix[i, j]:12.4f}", end='')
            print(f"  (sum={self.switching_matrix[i, :].sum():.4f})")


# Example usage
if __name__ == "__main__":
    choice = ContraceptiveChoice()
    
    print("Initial state:")
    choice.display_matrix()
    
    print("\n" + "="*60)
    print("Adding IUD method...")
    print("="*60)
    
    choice.add_method(
        year=2026, 
        method='iud',
        copy_from_row='pill',
        copy_from_col='pill',
        initial_share=0.1
    )

    # Alternative interfaces that should work but don't yet:
    choice.adjust_switching(
        year=2027,
        row='iud',
        col='pill',
        age_group='18-20',
        new_entry=0.15,
        renormalize=True
    )

    #
    
    print("\nAfter adding IUD:")
    choice.display_matrix()