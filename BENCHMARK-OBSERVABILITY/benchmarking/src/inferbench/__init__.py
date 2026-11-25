"""inferbench — vLLM benchmarking harness.

Runs vLLM as an imported library so the scheduler, KV cache and prefix cache
are readable from Python. See `PLAN.md` at the workspace root for the design.

Heavy imports are deliberately absent here: `import inferbench` must work on a
machine with no vLLM and no GPU, which is what keeps the test suite runnable
anywhere.
"""

__version__ = "0.1.0"
