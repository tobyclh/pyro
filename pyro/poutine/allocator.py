from __future__ import absolute_import, division, print_function

import contextlib


class TensorDimAllocator(object):
    def __init__(self):
        self._slots = []  # contains a site name if slot is used or None if slot is free

    def allocate(self, name):
        assert isinstance(name, str)
        # Try to reuse an existing slot.
        for slot, existing_name in enumerate(self._slots):
            if existing_name is None:
                self._slots[slot] = name
                return slot
        # Create a new slot.
        slot = len(self._slots)
        self._slots.append(name)
        return slot

    def free(self, slot):
        assert self._slots[slot] is not None, 'double free'
        self._slots[slot] = None

    @contextlib.contextmanager
    def local(self, name):
        slot = self.allocate(name)
        yield slot
        self.free(slot)
