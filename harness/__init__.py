"""Local research harness for the JED red-team competition.

Everything here is *our* tooling — never shipped in the submitted attack.py. It reuses
the installed aicomp_sdk scoring/predicate/cell/guardrail code directly (never
reimplements it) so our offline measurements match the evaluator exactly.
"""
