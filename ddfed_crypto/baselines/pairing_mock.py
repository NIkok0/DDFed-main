# DDFed/ddfed_crypto/baselines/pairing_mock.py
"""
Mock implementation of charm.toolbox.pairinggroup for development/testing.
This provides a simplified pairing-like interface without the actual cryptographic operations.
"""

import hashlib
import os
import random


class _OrderValue(int):
    """Int value that is also callable for compatibility."""

    def __call__(self):
        return int(self)


class MockPairingGroup:
    """Mock pairing group implementation"""
    
    def __init__(self, curve='SS512'):
        self.curve = curve
        # Keep both `group.order` and `group.order()` compatible.
        self.order = _OrderValue(2**256 - 2**224 - 2**96 + 2**64 - 1)  # P-256 order
    
    def init(self, group_type, value):
        """Initialize a group element"""
        # In multiplicative groups (G1/G2/GT), store exponent form internally.
        # External callers usually pass 1 for identity; map it to exponent 0.
        if group_type in ['G1', 'G2', 'GT'] and int(value) == 1:
            value = 0
        return MockGroupElement(group_type, value, self)
    
    def random(self, group_type):
        """Generate random group element"""
        if group_type == 'ZR':
            return random.randint(1, self.order - 1)
        elif group_type in ['G1', 'G2', 'GT']:
            return MockGroupElement(group_type, random.randint(1, self.order - 1), self)
        return random.randint(1, self.order - 1)
    
    def hash(self, data_str, group_type):
        """Hash string to group element"""
        h = hashlib.sha256(data_str.encode()).digest()
        val = int.from_bytes(h, 'big') % self.order
        if group_type == 'ZR':
            return MockGroupElement(group_type, val, self)
        # Avoid zero-exponent bases for non-trivial generators/hashes.
        return MockGroupElement(group_type, max(1, val), self)
    
    def serialize(self, element):
        """Serialize group element to bytes"""
        if isinstance(element, MockGroupElement):
            return element.value.to_bytes(32, 'big')
        return str(element).encode()
    
class MockGroupElement:
    """Mock group element"""
    
    def __init__(self, group_type, value, group):
        self.group_type = group_type
        self.value = value if isinstance(value, int) else int(value)
        self.group = group
    
    def __mul__(self, other):
        """Group multiplication (G1, G2, GT)"""
        if self.group_type in ['G1', 'G2', 'GT']:
            # Exponent representation: multiply groups => add exponents.
            other_val = other.value if isinstance(other, MockGroupElement) else int(other)
            new_val = (self.value + other_val) % self.group.order
            return MockGroupElement(self.group_type, new_val, self.group)

        if isinstance(other, MockGroupElement):
            new_val = (self.value * other.value) % self.group.order
            return MockGroupElement(self.group_type, new_val, self.group)
        new_val = (self.value * int(other)) % self.group.order
        return MockGroupElement(self.group_type, new_val, self.group)
    
    def __pow__(self, exponent):
        """Group exponentiation"""
        exp = int(exponent) if isinstance(exponent, MockGroupElement) else exponent
        if self.group_type in ['G1', 'G2', 'GT']:
            # Exponent representation: (g^a)^x = g^(a*x)
            new_val = (self.value * int(exp)) % self.group.order
            return MockGroupElement(self.group_type, new_val, self.group)
        new_val = pow(self.value, exp, self.group.order)
        return MockGroupElement(self.group_type, new_val, self.group)
    
    def __truediv__(self, other):
        """Group division (multiply by inverse)"""
        if self.group_type in ['G1', 'G2', 'GT']:
            # Exponent representation: divide groups => subtract exponents.
            other_val = other.value if isinstance(other, MockGroupElement) else int(other)
            new_val = (self.value - other_val) % self.group.order
            return MockGroupElement(self.group_type, new_val, self.group)

        if isinstance(other, MockGroupElement):
            inv = pow(other.value, -1, self.group.order)
            new_val = (self.value * inv) % self.group.order
            return MockGroupElement(self.group_type, new_val, self.group)
        inv = pow(int(other), -1, self.group.order)
        new_val = (self.value * inv) % self.group.order
        return MockGroupElement(self.group_type, new_val, self.group)
    
    def __int__(self):
        """Convert to integer"""
        return self.value
    
    def __repr__(self):
        return f"Mock{self.group_type}({self.value})"


# Global pairing group instance
_pairing_group = None


def PairingGroup(curve='SS512'):
    """Get or create pairing group"""
    global _pairing_group
    if _pairing_group is None:
        _pairing_group = MockPairingGroup(curve)
    return _pairing_group


# Group type constants
ZR = 'ZR'
G1 = 'G1'
G2 = 'G2'
GT = 'GT'


def pair(g1, g2):
    """Mock bilinear pairing function"""
    # Bilinear-compatible mock: e(g1^a, g2^b) = gt^(a*b).
    if isinstance(g1, MockGroupElement) and isinstance(g2, MockGroupElement):
        result = (g1.value * g2.value) % g1.group.order
        return MockGroupElement('GT', result, g1.group)
    return 1