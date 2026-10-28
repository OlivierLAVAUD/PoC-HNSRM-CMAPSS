# legacy/__init__.py
"""Modules kept for the synthetic demonstration.

They are **not** part of the C-MAPSS FD001 pipeline: their inputs
(``temperature``, ``vibration``, ``pressure``, ``efficiency_index``) belong to
the original synthetic example, not to FD001 sensors.
"""

LEGACY_MODULES = ('synthetic_data_loader', 'adaptive_controller')

__all__ = ['LEGACY_MODULES']
