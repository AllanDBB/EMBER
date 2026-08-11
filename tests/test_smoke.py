import subprocess
import sys


def test_importa_y_tiene_version():
    import ember

    assert ember.__version__ == "0.1.0"


def test_el_nucleo_no_importa_torch():
    """El código que va al robot no puede arrastrar el stack de laboratorio."""
    code = (
        "import sys, ember.core, ember.memories; "
        "prohibidos = {'torch', 'scipy', 'matplotlib', 'gymnasium'}; "
        "encontrados = prohibidos & set(sys.modules); "
        "sys.exit(f'importa {encontrados}' if encontrados else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
