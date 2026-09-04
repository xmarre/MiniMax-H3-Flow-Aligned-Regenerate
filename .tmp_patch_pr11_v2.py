from pathlib import Path

source_path = Path('.tmp_patch_pr11.py')
code = source_path.read_text()
needle = 'tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()\n'
replacement = needle + 'tracked = [name for name in tracked if not name.startswith(".tmp_patch_pr11") and "_tmp_finalize_pr11.yml" not in name]\n'
assert code.count(needle) == 1
code = code.replace(needle, replacement, 1)
exec(compile(code, str(source_path), 'exec'))
