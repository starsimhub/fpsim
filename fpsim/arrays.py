import numpy as np
import starsim as ss



class TwoDimensionalArr(ss.Arr):
    """
    A State that tracks a two-dimensional array of values, indexed by UIDs. Because this is a true State, it is linked to
    People and can grow dynamically as new agents are added. It is used for storing state that has multiple columns, such as
    the ages at which a child is born.

    It can be indexed by UIDs, slices, or boolean arrays, and supports setting values for new agents.

    example usage:
    self[uid]: returns the entire row of values for the specified uid
    self[uid, col]: returns the value in the specified column for the specified uid
    self[slice]: returns all rows for the specified slice
    self[bool_arr]: returns all rows where the boolean array is True

    Args:
        name (str): Name of the array
        dtype (type): Data type of the array (e.g., np.float64)
        default: Default value to use when setting new agents
        nan: Value to use for NaN entries
        label (str): Label for the array, used in plots and reports
        skip_init (bool): If True, do not initialize the array; useful for module state definitions

    """
    def __init__(self, name=None, dtype=None, default=None, nan=None, label=None, skip_init=False, people=None, ncols=1):
        # Set attributes
        self.name = name
        self.label = label or name
        self.default = default
        self.nan = nan
        self.dtype = dtype
        self.people = people # Used solely for accessing people.auids
        self.ncols = ncols

        if self.people is None:
            # This Arr is being defined in advance (e.g., as a module state) and we want a bidirectional link
            # with a People instance for dynamic growth. These properties will be initialized later when the
            # People/Sim are initialized
            self.len_used = 0
            self.len_tot = 0
            self.initialized = skip_init
            self.raw = np.empty((0,self.ncols), dtype=dtype)
        else:
            # This Arr is a temporary object used for intermediate calculations when we want to index an array
            # by UID (e.g., inside an update() method). We allow this state to reference an existing, initialized
            # People object, but do not register it for dynamic growth
            self.len_used = self.people.uid.len_used
            self.len_tot = self.people.uid.len_tot
            self.initialized = True
            self.raw = np.full(shape=(self.len_tot, self.ncols), dtype=self.dtype, fill_value=self.nan)

        return

    def _convert_key(self, key):
        """
        Used for getitem and setitem to determine whether the key is indexing
        the raw array (``raw``) or the active agents (``values``), and to convert
        the key to array indices if needed.
        """
        if isinstance(key, (ss.uids, int, ss.dtypes.int)):
            return key
        elif isinstance(key, (ss.BoolArr, ss.IndexArr)):
            return key.uids
        elif isinstance(key, slice):
            return self.auids[key]
        elif not np.isscalar(key) and len(key) == 0: # Handle [], np.array([]), etc.
            return ss.uids()
        elif isinstance(key, np.ndarray) and ss.options.reticulate:
            return key.astype(int)
        else:
            errormsg = f'Indexing an Arr ({self.name}) by ({key}) is ambiguous or not supported. Use ss.uids() instead, or index Arr.raw or Arr.values.'
            raise Exception(errormsg)

    def __getitem__(self, key):
        if isinstance(key, (slice, tuple)):
            index = self._convert_key(key[0])
            return self.raw[index, key[1]]
        else:
            index = self._convert_key(key)
            return self.raw[index, :]

    def __setitem__(self, key, value):
        if isinstance(key, (slice, tuple)):
            index = self._convert_key(key[0])
            self.raw[index, key[1]] = value
        return

    def __gt__(self, other): raise NotImplementedError("Comparison not supported for TwoDimensionalArray")
    def __lt__(self, other): raise NotImplementedError("Comparison not supported for TwoDimensionalArray")
    def __ge__(self, other): raise NotImplementedError("Comparison not supported for TwoDimensionalArray")
    def __le__(self, other): raise NotImplementedError("Comparison not supported for TwoDimensionalArray")
    def __eq__(self, other): raise NotImplementedError("Comparison not supported for TwoDimensionalArray")
    def __ne__(self, other): raise NotImplementedError("Comparison not supported for TwoDimensionalArray")

    def count(self):
        raise NotImplementedError("count() is not implemented for TwoDimensionalArray.")

    def set(self, uids, new_vals=None):
        """ Set the values for the specified UIDs"""
        if new_vals is None:
            if isinstance(self.default, ss.Dist):
                new_vals = self.default.rvs(uids)
            elif callable(self.default):
                new_vals = self.default(len(uids))
            elif self.default is not None:
                new_vals = self.default
            else:
                new_vals = self.nan
        self.raw[uids, :] = new_vals
        return new_vals

    @property
    def isnan(self):
        return self.asnew(np.isnan(self), cls=ss.BoolArr)

    @property
    def values(self):
        """ Return the values of the active agents """
        return self.raw[self.auids,:]

    def grow(self, new_uids=None, new_vals=None):
        """
        Add new agents to an Arr

        This method is normally only called via `People.grow()`.

        Args:
            new_uids: Numpy array of UIDs for the new agents being added
            new_vals: If provided, assign these state values to the new UIDs
        """
        orig_len = self.len_used
        n_new = len(new_uids)
        self.len_used += n_new  # Increase the count of the number of agents by `n` (the requested number of new agents)

        # Physically reshape the arrays, if needed
        if orig_len + n_new > self.len_tot:
            n_grow = max(n_new, self.len_tot//2)  # Minimum 50% growth, since growing arrays is slow
            new_empty = np.empty((n_grow, self.ncols), dtype=self.dtype) # 10x faster than np.zeros()
            self.raw = np.concatenate([self.raw, new_empty], axis=0)
            self.len_tot = len(self.raw)
            if n_grow > n_new: # We added extra space at the end, set to NaN
                nan_uids = np.arange(self.len_used, self.len_tot)
                self.set_nan(nan_uids)

        # Set new values, and NaN if needed
        self.set(new_uids, new_vals=new_vals) # Assign new default values to those agents
        return