import logging
from types import ModuleType
from importlib import reload, import_module
from IPython.core.magic import register_line_magic, register_cell_magic

logger = logging.getLogger(__name__)


def _reload(module, reload_all, reloaded):
    if isinstance(module, ModuleType):
        module_name = module.__name__
    elif isinstance(module, str):
        module_name, module = module, import_module(module)
    else:
        raise TypeError(
            "'module' must be either a module or str; "
            f"got: {module.__class__.__name__}")

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        check = (
            # is it a module?
            isinstance(attr, ModuleType)

            # has it already been reloaded?
            and attr.__name__ not in reloaded

            # is it a proper submodule? (or just reload all)
            and (reload_all or attr.__name__.startswith(module_name))
        )
        if check:
            _reload(attr, reload_all, reloaded)

    logger.debug(f"reloading module: {module.__name__}")
    reload(module)
    reloaded.add(module_name)


def reload_recursive(module, reload_external_modules=False):
    """
    Recursively reload a module (in order of dependence).
    Parameters
    ----------
    module : ModuleType or str
        The module to reload.
    reload_external_modules : bool, optional
        Whether to reload all referenced modules, including external ones which
        aren't submodules of ``module``.
    """
    _reload(module, reload_external_modules, set())


@register_line_magic('reload')
def reload_magic(module):
    """
    Reload module on demand.
    Examples
    --------
    # >>> %reload my_module
    reloading module: my_module
    # >>> import my_module  # import (again) after reload
    """
    reload_recursive(module)
