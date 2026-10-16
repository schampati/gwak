"""Backward-compatible import/CLI for sky_position_antenna_patterns.py."""
if __package__:
    from . import sky_position_antenna_patterns as _implementation
else:
    import sky_position_antenna_patterns as _implementation

# Preserve existing public and private imports used by archived scripts/tests.
for _name in dir(_implementation):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_implementation, _name)

if __name__ == '__main__':
    main()
