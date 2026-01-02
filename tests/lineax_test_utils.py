"""Utilities for importing and using Lineax test functions."""
import sys
import importlib.util as iplu
from pathlib import Path
from typing import Any


def _load_lineax_test_module(filename: str) -> Any:
    """
    Load a module from lineax/tests using importlib.
    
    Args:
        filename: Filename in lineax/tests (e.g., 'helpers.py')
    
    Returns:
        The loaded module
    """
    module_path = Path(__file__).parent.parent / "lineax" / "tests" / filename
    module_name = f"lineax.tests.{module_path.stem}"
    
    spec = iplu.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {module_path}")
    
    module = iplu.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def _build_lineax_tests_class():
    """
    Dynamically discover and load all test modules from lineax/tests,
    creating a namespace class with all test functions and helpers.
    """
    lineax_tests_dir = Path(__file__).parent.parent / "lineax" / "tests"
    
    # Find all Python files in lineax/tests
    test_files = sorted(lineax_tests_dir.glob("*.py"))
    
    class LineaxTests:
        """Namespace for lineax test utilities and functions."""
        pass
    
    # Create a subnamespace for storing modules
    class _modules:
        """Namespace for storing the actual test modules."""
        pass
    
    LineaxTests._modules = _modules
    
    # Load each module and extract its functions
    for test_file in test_files:
        filename = test_file.name
        
        # Skip special files
        if filename.startswith("_"):
            continue
        
        try:
            module = _load_lineax_test_module(filename)
            
            # Add all public attributes to LineaxTests
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                
                attr = getattr(module, attr_name)
                
                # Skip imported modules to avoid pollution
                if hasattr(attr, "__module__") and attr.__module__ != module.__name__:
                    # This is an import from another module, skip it
                    # Exception: keep it if it's from lineax itself
                    if not attr.__module__.startswith("lineax"):
                        continue
                
                # Add callables as staticmethods, everything else as-is
                if callable(attr):
                    setattr(LineaxTests, attr_name, staticmethod(attr))
                else:
                    setattr(LineaxTests, attr_name, attr)
            
            # Store the module itself under _modules to avoid name collisions
            setattr(LineaxTests._modules, test_file.stem, module)
            
        except Exception as e:
            # Log but don't fail if a module can't be loaded
            print(f"Warning: Could not load {filename}: {e}")
            continue
    
    return LineaxTests


# Create the class on import
lx_tests = _build_lineax_tests_class()
