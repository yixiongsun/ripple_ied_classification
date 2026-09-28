"""Compatibility imports for the consolidated event-classification module."""

from .event_classification import detect_ieds, export_ieds


# Historical notebook name; prefer ``detect_ieds`` in new code.
find_ripples = detect_ieds
