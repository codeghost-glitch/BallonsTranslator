# modified from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py

import importlib
import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List


class LazyModuleError(Exception):
    pass


@dataclass
class ModuleSpec:
    """Static module metadata used by the UI before importing heavy modules.

    Example:
        >>> class DemoModule:
        ...     pass
        >>> spec = ModuleSpec(
        ...     key='demo',
        ...     import_path='unused',
        ...     class_name='DemoModule',
        ...     params={'device': {'value': 'cpu'}},
        ...     resolved_class=DemoModule,
        ... )
        >>> spec.resolve() is DemoModule
        True
        >>> copied = spec.params_copy()
        >>> copied == spec.params
        True
        >>> copied is spec.params
        False
    """

    key: str
    import_path: str
    class_name: str
    module_type: str = ''
    params: Dict = None
    download_file_list: List = None
    download_file_on_load: bool = False
    dependencies: List[str] = field(default_factory=list)
    supported_src_list: List[str] = None
    supported_tgt_list: List[str] = None
    available: bool = True
    availability_error: str = ''
    resolved_class: type = None
    metadata_warnings: List[str] = field(default_factory=list)

    def resolve(self):
        # Import the concrete module only after the user selects it.
        if self.resolved_class is not None:
            return self.resolved_class
        if not self.available:
            raise LazyModuleError(self.availability_error or f'{self.key} is not available on this platform.')
        try:
            module = importlib.import_module(self.import_path)
            self.resolved_class = getattr(module, self.class_name)
            setattr(self.resolved_class, '_module_spec', self)
            return self.resolved_class
        except ModuleNotFoundError as e:
            missing = e.name or str(e)
            raise LazyModuleError(
                f'Module "{self.key}" requires Python package "{missing}". '
                f'Install the dependency before selecting this module.'
            ) from e
        except Exception as e:
            raise LazyModuleError(f'Failed to import module "{self.key}" from {self.import_path}: {e}') from e

    def params_copy(self):
        return deepcopy(self.params)

    @property
    def name(self):
        return self.key

class Registry:
    """A registry to map strings to classes or lazy ModuleSpecs.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register_module()
        >>> class ResNet:
        >>>     pass

    Args:
        name (str): Registry name.
    """

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """Get the registry record for ``key``, or ``None``.

        Args:
            key (str): The class name in string format.

        Returns:
            class | ModuleSpec | None: The corresponding record.
        """
        return self._module_dict.get(key)

    def _register_module(self, module_class, module_name=None, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, '
                            f'but got {type(module_class)}')

        if module_name is None:
            module_name = module_class.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
            
        for name in module_name:
            existing = self._module_dict.get(name)
            if isinstance(existing, ModuleSpec):
                # A lazy spec can be replaced by the real class after import.
                existing.resolved_class = module_class
                setattr(module_class, '_module_spec', existing)
                self._module_dict[name] = module_class
                continue
            if not force and name in self._module_dict:
                raise KeyError(f'{name} is already registered '
                               f'in {self.name}')
            self._module_dict[name] = module_class

    def register_lazy_module(self, spec: ModuleSpec, force=False):
        if not isinstance(spec, ModuleSpec):
            raise TypeError(f'spec must be a ModuleSpec, but got {type(spec)}')
        existing = self._module_dict.get(spec.key)
        if not force and existing is not None:
            if isinstance(existing, ModuleSpec) and existing.import_path == spec.import_path and existing.class_name == spec.class_name:
                return spec
            if inspect.isclass(existing) and existing.__name__ == spec.class_name:
                spec.resolved_class = existing
                setattr(existing, '_module_spec', spec)
                return spec
            raise KeyError(f'{spec.key} is already registered in {self.name}')
        self._module_dict[spec.key] = spec
        return spec

    def resolve_module(self, key):
        module = self.get(key)
        if isinstance(module, ModuleSpec):
            # Keep the resolved class cached so later access is normal registry use.
            resolved = module.resolve()
            self._module_dict[key] = resolved
            return resolved
        return module

    def get_spec(self, key):
        module = self.get(key)
        if isinstance(module, ModuleSpec):
            return module
        if inspect.isclass(module):
            spec = getattr(module, '_module_spec', None)
            if isinstance(spec, ModuleSpec):
                spec.resolved_class = module
                return spec
            return ModuleSpec(
                key=key,
                import_path=module.__module__,
                class_name=module.__name__,
                params=deepcopy(getattr(module, 'params', None)),
                download_file_list=deepcopy(getattr(module, 'download_file_list', None)),
                download_file_on_load=getattr(module, 'download_file_on_load', False),
                dependencies=deepcopy(getattr(module, 'dependencies', [])),
                metadata_warnings=[],
                resolved_class=module,
            )
        return None

    def register_module(self, name=None, force=False, module=None):
        """Register a module.

        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass

            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)

        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                'name must be either of None, an instance of str or a sequence'
                f'  of str, but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            
            self._register_module(
                module_class=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force)
            return cls

        return _register
    
    def __getitem__(self, key: str):
        return self.get(key)
