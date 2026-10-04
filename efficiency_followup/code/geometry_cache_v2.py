"""Exact, model-local memoization of original geometry-only internal banks.

No physical solve or domain-bank algorithm is implemented here. The first
value is acquired by the original ctx._internal. Reuse requires same model,
same GD object/content, same grid/k/volume signature, count, seed and callable.
Hash construction/comparison, storage, copies and invalidation are charged.
V2 fixes v1 unstable marshal-based callable keys. This helper has NOT been admitted to a physical deployment experiment.
"""
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
from types import CodeType
from numbers import Integral
from pathlib import Path
import time
import numpy as np


CODE_FINGERPRINT_SCHEMA='a17.python_code_semantics.v2'


def _constant_semantics(value):
    """Canonical constants only; never object repr, marshal or mutable runtime state."""
    if value is None:return ['none']
    if value is Ellipsis:return ['ellipsis']
    if type(value) is bool:return ['bool',value]
    if type(value) is int:return ['int',str(value)]
    if type(value) is float:return ['float',value.hex()]
    if type(value) is complex:return ['complex',value.real.hex(),value.imag.hex()]
    if type(value) is str:return ['str',value]
    if type(value) is bytes:return ['bytes',value.hex()]
    if type(value) is tuple:return ['tuple',[_constant_semantics(v) for v in value]]
    if type(value) is frozenset:
        entries=[_constant_semantics(v) for v in value]
        return ['frozenset',sorted(entries,key=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True))]
    if isinstance(value,CodeType):return ['code',_code_semantics(value)]
    raise TypeError('unsupported code constant type: '+type(value).__name__)


def _code_semantics(code):
    # Public original bytecode/constant tables remain stable after execution.
    # Exclude filename/line tables and marshal reference/intern bookkeeping.
    # Argument layout, variable binding and exception behavior are retained.
    return dict(schema=CODE_FINGERPRINT_SCHEMA,co_code=code.co_code.hex(),
        co_consts=[_constant_semantics(v) for v in code.co_consts],co_names=list(code.co_names),
        co_varnames=list(code.co_varnames),co_freevars=list(code.co_freevars),co_cellvars=list(code.co_cellvars),
        co_argcount=code.co_argcount,co_posonlyargcount=getattr(code,'co_posonlyargcount',0),
        co_kwonlyargcount=code.co_kwonlyargcount,
        co_exceptiontable=getattr(code,'co_exceptiontable',b'').hex())


def _callable_identity(fn):
    code=getattr(fn,'__code__',None)
    source=inspect.getsourcefile(fn)
    payload=json.dumps(_code_semantics(code),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('ascii') if code is not None else None
    return dict(module=getattr(fn,'__module__',None),qualname=getattr(fn,'__qualname__',type(fn).__name__),
                code_fingerprint_schema=CODE_FINGERPRINT_SCHEMA,
                code_semantics_sha256=hashlib.sha256(payload).hexdigest() if payload is not None else None,
                source_name=Path(source).name if source else None,
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None)


class InstalledInternalHook:
    """Restore exactly the original callable, only while this hook still owns it."""
    def __init__(self,ctx,original,wrapped):self.ctx,self.original,self.wrapped=ctx,original,wrapped
    def restore(self):
        if self.ctx._internal is not self.wrapped:raise RuntimeError('another owner changed ctx._internal')
        self.ctx._internal=self.original


class GeometryInternalBankCache:
    """Single-model, bounded bank store. Changes invalidate, never approximate.

    Safe only for the audited original internal mode provider whose outputs
    depend exclusively on model.GD/count/seed. Caller supplies provenance;
    installation itself does not establish that scientific precondition.
    """
    def __init__(self,*,max_entries=8):
        if not isinstance(max_entries,int) or max_entries<1:raise ValueError('positive cache bound required')
        self.max_entries=max_entries;self._model=None;self._GD=None;self._signature=None
        self._entries={};self.records=[];self.installations=[]
        self.acquisitions=self.hits=self.invalidations=0

    @contextmanager
    def _stage(self,record,name):
        started=time.perf_counter()
        try:yield
        finally:record['exclusive_stage_wall_s'][name]=record['exclusive_stage_wall_s'].get(name,0.)+time.perf_counter()-started

    def install(self,ctx,model,*,audited_provider_sha256):
        """Replace one newly constructed ctx._internal with an explicit wrapper.

        audited_provider_sha256 is an externally hash-locked evidence identity,
        not a self-issued certificate. It participates in each bank key.
        """
        started=time.perf_counter()
        if not isinstance(audited_provider_sha256,str) or len(audited_provider_sha256)!=64 or any(c not in '0123456789abcdef' for c in audited_provider_sha256):
            raise ValueError('audited provider requires its locked SHA256')
        original=ctx._internal
        if getattr(original,'_a17_geometry_cache_wrapper',False):raise ValueError('do not recursively wrap a cache hook')
        identity=_callable_identity(original)
        hook_geometry=dict(model=model,GD=None,signature=None)
        def memoized(count,seed):
            return self._get(ctx,model,original,identity,audited_provider_sha256,count,seed,hook_geometry)
        memoized._a17_geometry_cache_wrapper=True
        memoized.__name__='a17_geometry_memoized_internal'
        memoized.__module__=__name__
        ctx._internal=memoized
        installation=dict(semantic_mode='geometry_memoization',callable_replacement=True,numerical_monkeypatch=True,
                          numerical_algorithm_changed=False,actual_callables=dict(original=identity,installed=_callable_identity(memoized)),
                          audited_provider_sha256=audited_provider_sha256,install_wall_s=time.perf_counter()-started,
                          scope='Only original geometry internal bank V/meta; no L/B/state/Jd/pool/anchor/policy storage')
        self.installations.append(installation)
        self._ledger(ctx,'geometry_cache_install_wall_s',installation['install_wall_s'])
        return InstalledInternalHook(ctx,original,memoized)

    @staticmethod
    def _ledger(ctx,key,value):
        ledger=getattr(ctx,'ledger',None)
        if ledger is not None and hasattr(ledger,'add'):ledger.add(key,value)

    def _geometry_signature(self,model,record):
        with self._stage(record,'geometry_signature_hash'):
            GD=model.GD
            if not isinstance(GD,np.ndarray) or GD.ndim!=2 or GD.dtype!=np.complex128 or GD.shape!=(int(model.n),int(model.n)):
                raise ValueError('audited GD must be complex128 n by n ndarray')
            points=np.asarray(model.points)
            if points.dtype!=np.float64 or points.ndim!=2 or points.shape!=(int(model.N),3):
                raise ValueError('audited grid must be original FP64 N by 3 points')
            # memoryview avoids a full bytes allocation for contiguous storage.
            gd_view=np.ascontiguousarray(GD);point_view=np.ascontiguousarray(points)
            record['hash_input_copy_bytes']+=0 if np.shares_memory(gd_view,GD) else gd_view.nbytes
            record['hash_input_copy_bytes']+=0 if np.shares_memory(point_view,points) else point_view.nbytes
            record['hash_read_bytes']+=gd_view.nbytes+point_view.nbytes
            gd_hash=hashlib.sha256(memoryview(gd_view).cast('B')).hexdigest()
            points_hash=hashlib.sha256(memoryview(point_view).cast('B')).hexdigest()
            signature=(int(model.n),int(model.N),float(model.k).hex(),float(model.volume).hex(),
                       GD.shape,GD.dtype.str,gd_hash,points.shape,points.dtype.str,points_hash)
        return GD,signature

    def _invalidate(self,record,reason):
        with self._stage(record,'cache_invalidation'):
            discarded=len(self._entries);storage=sum(v['storage_bytes'] for v in self._entries.values())
            self._entries.clear();self.invalidations+=1
            record['invalidations'].append(dict(reason=reason,discarded_banks=discarded,discarded_array_bytes=storage))

    def _get(self,ctx,model,original,identity,provider,count,seed,hook_geometry):
        started=time.perf_counter()
        record=dict(semantic_mode='geometry_memoization',count=int(count) if isinstance(count,Integral) else None,
                    seed=int(seed) if isinstance(seed,Integral) else None,branch='PENDING',status='OK',
                    exclusive_stage_wall_s={},hash_read_bytes=0,hash_input_copy_bytes=0,
                    bank_storage_copy_bytes=0,return_copy_bytes=0,invalidations=[],
                    physical_GD_operations_on_hit=0,audited_provider_sha256=provider)
        result=None
        try:
            GD,signature=self._geometry_signature(model,record)
            with self._stage(record,'identity_and_content_comparison'):
                model_changed=self._model is not model
                gd_changed=self._GD is not GD
                geometry_changed=self._signature!=signature
            if model_changed or gd_changed or geometry_changed:
                self._invalidate(record,'new_model' if model_changed else 'GD_identity_changed' if gd_changed else 'geometry_or_GD_content_changed')
                self._model,self._GD,self._signature=model,GD,signature
            if hook_geometry['signature'] is not None and (hook_geometry['GD'] is not GD or hook_geometry['signature']!=signature):
                record['branch']='OLD_CONTEXT_GEOMETRY_MUTATION_REJECTED'
                raise ValueError('GD/grid changed after this context was used; original context has its own local internal cache; construct a fresh context')
            hook_geometry.update(GD=GD,signature=signature)
            if not isinstance(count,Integral) or not isinstance(seed,Integral):
                # Preserve original argument/error semantics; never coerce a
                # previously invalid call into a new successful cache request.
                record['branch']='UNCACHEABLE_ARGUMENT_ORIGINAL_CALL'
                with self._stage(record,'original_provider_call'):result=original(count,seed)
                return result
            key=(int(count),int(seed),provider,identity['module'],identity['qualname'],identity['code_fingerprint_schema'],identity['code_semantics_sha256'],identity['source_sha256'])
            with self._stage(record,'bank_key_lookup'):entry=self._entries.get(key)
            if entry is None:
                record['branch']='MISS_ORIGINAL_ACQUISITION';self.acquisitions+=1
                with self._stage(record,'original_provider_call'):value=original(count,seed)
                if not isinstance(value,tuple) or len(value)!=2:
                    raise ValueError('audited provider must return original (V,metadata) pair')
                V,metadata=value
                if not isinstance(V,np.ndarray) or V.dtype!=np.complex128 or V.ndim!=2 or V.shape[0]!=model.n or not isinstance(metadata,dict):
                    raise ValueError('audited original V/meta contract differs')
                # Rehash after acquisition too: detect concurrent mutation rather
                # than caching a result for a pre-call GD that may no longer exist.
                after_GD,after_signature=self._geometry_signature(model,record)
                if after_GD is not GD or after_signature!=signature:
                    self._invalidate(record,'geometry_mutated_during_acquisition')
                    self._GD=after_GD;self._signature=after_signature
                    record['branch']='CHANGED_DURING_ACQUISITION_NOT_STORED'
                    return value
                if len(self._entries)>=self.max_entries:
                    with self._stage(record,'bounded_bank_eviction'):
                        oldkey=next(iter(self._entries));old=self._entries.pop(oldkey)
                        record['invalidations'].append(dict(reason='entry_bound',discarded_banks=1,discarded_array_bytes=old['storage_bytes']))
                with self._stage(record,'first_bank_storage_copy'):
                    stored=V.copy();stored.setflags(write=False)
                    entry=dict(V=stored,metadata=copy.deepcopy(metadata),storage_bytes=stored.nbytes)
                    self._entries[key]=entry;record['bank_storage_copy_bytes']=stored.nbytes
                # Original first result is returned unchanged; storing a private
                # copy cannot alter the original acquisition/counter semantics.
                result=value
            else:
                record['branch']='EXACT_GEOMETRY_BANK_HIT';self.hits+=1
                with self._stage(record,'hit_bank_and_metadata_copy'):
                    result=(entry['V'].copy(),copy.deepcopy(entry['metadata']))
                    record['return_copy_bytes']=result[0].nbytes
            record['persistent_array_bytes']=sum(v['storage_bytes'] for v in self._entries.values())
            record['metadata_storage_bytes']='not measured; Python metadata object overhead excluded from explicit array bytes'
            return result
        except BaseException as error:
            record['status']='ERROR';record['error_type']=type(error).__name__;raise
        finally:
            record['wall_s']=time.perf_counter()-started
            measured=sum(record['exclusive_stage_wall_s'].values())
            if measured>record['wall_s']+1e-9:raise RuntimeError('exclusive cache times exceed actual call wall')
            record['unattributed_wall_s']=max(0.,record['wall_s']-measured)
            record['wall_scope']='Hash/check/acquisition/copy/invalidation and wrapper arithmetic; excludes final receipt publication, charged by enclosing job wall.'
            self.records.append(record)
            for key,value in [('geometry_cache_calls',1),('geometry_cache_hits',int(record['branch']=='EXACT_GEOMETRY_BANK_HIT')),
                              ('geometry_cache_hash_read_bytes',record['hash_read_bytes']),
                              ('geometry_cache_copy_bytes',record['hash_input_copy_bytes']+record['bank_storage_copy_bytes']+record['return_copy_bytes']),
                              ('geometry_cache_wall_s',record['wall_s'])]:self._ledger(ctx,key,value)

    def report(self):
        return dict(semantic_mode='geometry_memoization',numerical_monkeypatch=True,
                    actual_callables=self.installations,acquisitions=self.acquisitions,hits=self.hits,
                    invalidations=self.invalidations,persistent_array_bytes=sum(v['storage_bytes'] for v in self._entries.values()),
                    max_entries=self.max_entries,records=copy.deepcopy(self.records),physical_admission=False,
                    cache_scope='One model and exact audited geometry bank outputs only; no state/L/B/Jd/pool/anchor/policy reuse')
