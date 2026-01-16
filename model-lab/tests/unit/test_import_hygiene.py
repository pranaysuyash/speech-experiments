import sys
import unittest
from pathlib import Path

class TestImportHygiene(unittest.TestCase):
    def test_import_roots(self):
        """
        Verify that key packages import from the local repository, NOT site-packages.
        This prevents 'shadowing' where you edit code but run an installed version.
        """
        repo_root = Path(__file__).parent.parent.parent.resolve()
        
        # List of top-level packages to check
        packages_to_check = ["harness"]
        
        for pkg_name in packages_to_check:
            # Force reload if already imported to ensure we check current resolution
            if pkg_name in sys.modules:
                del sys.modules[pkg_name]
                
            module = __import__(pkg_name)
            
            # Get path
            if hasattr(module, "__file__") and module.__file__:
                mod_path = Path(module.__file__).resolve()
                
                # Check if it is under repo root
                if repo_root not in mod_path.parents:
                    self.fail(
                        f"Import Hygiene Failure: '{pkg_name}' resolved to '{mod_path}', "
                        f"which is NOT under repo root '{repo_root}'.\n"
                        f"Action: Uninstall the package from your environment or reinstall as editable (`uv pip install -e .`)"
                    )
            else:
                print(f"Warning: {pkg_name} has no __file__, cannot verify path.")

if __name__ == "__main__":
    unittest.main()
