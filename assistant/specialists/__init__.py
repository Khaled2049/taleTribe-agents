"""Specialists the director can consult: one bounded model call each.

Deliberately empty: ``assistant.tools`` imports ``specialists.base`` for the id
literals, so anything imported here would form a cycle back through tools.
"""
