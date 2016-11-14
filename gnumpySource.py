# ------------------------------------------------------------------------------- module init & shutdown

import numpy, operator, sys as _sys, types as types, time as _time, os as _os, __builtin__, collections as _collections, pdb as _pdb, gc as _gc, ctypes as _ctypes

useGpu = _os.environ.get('GNUMPY_USE_GPU', 'auto')
assert useGpu in t3('auto', 'yes', 'no'), "environment variable GNUMPY_USE_GPU, if present, should be one of 'auto', 'yes', 'no'."
if useGpu == 'auto':
 try: import cudamat as _cudamat; useGpu = 'yes'
 except: print 'gnumpy: failed to import cudamat. Using npmat instead. No GPU will be used.'; useGpu = 'no'
if useGpu == 'yes':
 import cudamat as _cudamat
elif useGpu == 'no':
 import npmat as _cudamat
 _precision = _os.environ.get('GNUMPY_CPU_PRECISION', '32')
 assert _precision in t3('32', '64', '128'), 'environment variable GNUMPY_CPU_PRECISION, if present, should have value 32, 64, or 128.'
 _cudamat.__DTYPE__ = eval('numpy.float'+_precision)

cmType = _cudamat.CUDAMatrix
isTijmen = False
if hasattr(_cudamat, 'ct'): ctInt = _cudamat.ct.c_int

def board_id_to_use():
 try:
  import gpu_lock
  return gpu_lock.obtain_lock_id()
 except:
  print 'gnumpy: failed to use gpu_lock. Using board #0 without knowing whether it is in use or not.'
  return 0
 
boardId = None
def init_gpu():
 """ picks a board and claims it (if using cudamat aot npmat). exception if there is no board. """
 if '__gpu_inited' in globals(): return
 global boardId
 if useGpu=='yes':
  boardId = ( board_id_to_use() if callable(board_id_to_use) else board_id_to_use)
  if boardId==-1: raise Exception('No gpu board is available. gnumpy will not function. Consider telling it to run on the CPU by setting environment variable GNUMPY_USE_GPU to "no".')
  _cudamat.cuda_set_device(boardId)
  _cudamat.cublas_init()
 _cudamat.CUDAMatrix.init_random(0)
 globals()['__gpu_inited'] = None

expensive_check_probability = 1 
acceptable_number_types = 'anything goes' # alternatives: 'no nans'; 'no nans or infs'; or a number indicating the max allowed abs
dont_check_number_types_in_non_garrays = True
class GnumpyNumberTypeException(Exception): pass

checking_number_type_now = False
def check_number_types(x):
 """ does some checks, and then returns x. """
 if acceptable_number_types == 'anything goes': return x # this is the typical case, and in this case I just want to leave this checking function asap.

 global checking_number_type_now
 if dont_check_number_types_in_non_garrays and not is_garray(x): return x
 if checking_number_type_now: return x # to prevent checks upon checks upon checks (infinite recursion)
 try:
  checking_number_type_now = True
  if acceptable_number_types == 'no nans': raise NotImplementedError
  elif acceptable_number_types == 'no nans or infs':
   if not garray(x, copy=False).all_real(): raise GnumpyNumberTypeException('Found values that violate the rule set by gnumpy.acceptable_number_types: "%s"' % acceptable_number_types)
  elif isNumber(acceptable_number_types):
   if (abs(garray(x, copy=False)) > acceptable_number_types).any2(): raise GnumpyNumberTypeException('Found values that violate the rule set by gnumpy.acceptable_number_types: "%s"' % acceptable_number_types)
  else: assert False, 'gnumpy: the value of variable "acceptable_number_types" must be one of "anything goes", "no nans", "no nans or infs".'
 finally:
  checking_number_type_now = False
 return x  
 


# ------------------------------------------------------------------------------- helpers copied from other files

def isFullSlice(x): return isSlice(x) and x == slice(None) # the first check is necessary to avoid returning a broadcast array of False's if x is an array
def isSequence(x): return isList(x) or isTuple(x) or type(x)==xrange
def insertT(tup, index, tupleToInsert): return tuple(tup[:index]) + tuple(tupleToInsert) + tuple(tup[index:])
def modifyT(tup, index, newValue): return tuple(tup[:index]) + t1(newValue) + tuple(tup[index+1:])
def deleteT(tup, start, end): return tup[:start] + tup[end:]
def prodT(x): return reduce(operator.mul, x, 1)
def findIndex3(tupOrGenerator): return findIndex2(tuple(tupOrGenerator), echo)
def isNumber(x): return type(x) in numberTypes
def nonSeqAsS(x): return ( x if isSequence(x) else t1(x))
t0=()
def reduceAdd(x): return reduce(operator.add, x)

def deleteT2(tup, index):
 index %= len(tup)
 return tup[:index] + tup[index+1:]

intTypes = set((types.IntType, types.LongType, numpy.int16, numpy.int32, numpy.int8, numpy.int64))
floatTypes = set((types.FloatType, numpy.float64, numpy.float32, getattr(numpy, 'float128', numpy.float64), getattr(numpy, 'float96', numpy.float64))) # considering numpy.float64 a number is debatable. it really is a numpy object, and behaves that way, too: it has a __mul__ which prevents garray.__rmul__ from getting the task. However, for most purposes it's a number.
numberTypes = intTypes | floatTypes
 
def allTheSame(tup):
 tup = tuple(tup)
 if len(tup)<=1: return True
 for elt in tup[1:]:
  if elt != tup[0]: return False
 return True





# ------------------------------------------------------------------------------- gnumpy specific helpers

def all2_(t, pred): return allT(mapTSimplified(t, pred))
def any2_(t, pred): return anyT(mapTSimplified(t, pred))

def doExpensiveCheck(): return numpy.random.rand() < expensive_check_probability

def as_garray(x): return ( x if is_garray(x) else garray(x))
def as_garray_or_scalar(x): return ( x if isNumber(x) or is_garray(x) else garray(x))
def as_numpy_array(x): return ( x.as_numpy_array() if is_garray(x) else numpy.array(x))

def cm_reshape(cm, newShape):
 if prodT(newShape)==0: return cm
 else: return cm.reshape(reverseT(newShape))

def cm_col_slice_write(cm, start, end, sourceCm):
 cm.set_row_slice(start, end, sourceCm)

def cm_col_slice_read(cm, start, end, target):
 cm.get_row_slice(start, end, target)
 return target

def cm_row_slice_read(cm, start, end):
 if start==end: return new_cm(t2(0, cm.shape[0])) # cudamat special case workaround
 if cm.shape[1]==1 and start==0 and end==1: return cm # cudamat special case workaround
 ret = cm.get_col_slice(start, end)
 return ret

def read_single_index(index, axisLen):
 index = int(index)
 if index>=axisLen or index<-axisLen: raise IndexError('index out of bounds. index %d requested on an axis of length %d' % t2(index, axisLen))
 return index % axisLen

def short_slice(i): return slice(i, i+1)

def read_simple_slice(sl, axisLen):
 assert sl.step in t2(None, 1), 'simple slice not understood'
 sFrom, sTo = slice(( None if sl.start==None else int(sl.start)), ( None if sl.stop==None else int(sl.stop))).indices(axisLen)[:2]
 if sFrom>sTo: sTo = sFrom
 return t3(sFrom, sTo, sTo-sFrom)

def extend_shape(shape, nAxes): return t1(1) * (nAxes-len(shape)) + shape
 
def cudamatHas(name):
 if not hasattr(_cudamat, '_cudamat'): return False
 return hasattr(_cudamat._cudamat, name)


# ------------------------------------------------------------------------------- memory management

max_memory_usage = numpy.inf # public

cmsForReuse = _collections.defaultdict(list) # dict from size to list of reusable (abandoned) cms
memoryInUse = 0
memoryUsers = _collections.defaultdict(lambda: t2(0, 0))
track_memory_usage = False

def new_cm(sizeOrShape):
 """
 Internal.
 Returns a new CUDAMatrix object of the given size.
 This is the only proc that allocs gpu mem.
 """
 global memoryInUse
 if isTuple(sizeOrShape):
  if prodT(sizeOrShape)==0: return new_cm(1) # cudamat workaround: cudamat can't handle size 0 arrays
  else: return new_cm(sizeOrShape[0]*sizeOrShape[1]).reshape(t2(sizeOrShape[1], sizeOrShape[0]))
 size = sizeOrShape
 if size==0: return _cudamat.empty(t2(1, 1)) # cudamat workaround
 if len(cmsForReuse[size])!=0:
  return cm_reshape(cmsForReuse[size].pop(), t2(1, size)) # re-use an abandoned cm
 init_gpu()
 if memoryInUse+size*4*5 > max_memory_usage: free_reuse_cache(False) # if we're somewhat close to the limit, then free what's easy to free, and hope that there are contiguous blocks available.
 if memoryInUse+size*4 > max_memory_usage: # if we're (still) OVER the limit, then do whatever can be done to make more mem available
  free_reuse_cache(True) # gc.collect can take quite some time
  if memoryInUse+size*4 > max_memory_usage:
   raise MemoryError('Gnumpy ran out of memory. Currently in use are %s; the maximum allowed is %s; so the present request for %s is refused. Free some memory and try again.' % t3(n_bytes_str(memoryInUse), n_bytes_str(max_memory_usage), n_bytes_str(size*4)))
 try:
  ret = _cudamat.empty(t2(size, 1))
  memoryInUse += size*4 # do this only if the malloc succeeded
  return ret
 except _cudamat.CUDAMatException, e: # this means that malloc failed
  raise MemoryError('The GPU failed to allocate the requested %d bytes of memory. This doesn\'t mean that your program is using too much memory. It does, however, mean that you should reduce the value of gnumpy.max_memory_usage (currently %s), to always have some memory unused (which is necessary to find contiguous large blocks of memory to allocate). Failing to allocate enough memory makes the GPU feel very unwell, so you are advised to restart Python now, or expect to see incoherent error messages and risk causing more serious damage.' % t2(size*4, str(max_memory_usage)))

def free_reuse_cache(completely=True):
 """
 This frees all GPU memory that is not in use but is kept allocated for re-use.
 If <completely> is set to False, this works quicker but less thoroughly.
 """
 if completely: _gc.collect() # this has to happen before the loop, because this may add more entries in cmsForReuse which then have to be freed by the loop
 global memoryInUse
 for size in cmsForReuse:
  while cmsForReuse[size]:
   cmsForReuse[size].pop()
   memoryInUse -= size*4
 del _gc.garbage[:] # this shouldn't be necessary at all, but for some reason perfectly referenced AND perfectly deletable cms get put there

def n_bytes_str(n):
 def base(s): return ( base(s[:-3]) + ',' + s[-3:] if len(s)>=4 else s)
 return base(str(n)) + ' bytes'
 
def memory_in_use(in_megabytes=False):
 """ returns the number of bytes (or megabytes if you asked for that) of GPU memory that are in use. """
 return memoryInUse // ( 2**20 if in_megabytes else 1)
   
def memory_available(free_reuse_cache_first):
 if free_reuse_cache_first: free_reuse_cache()
 return max_memory_usage - memory_in_use()

def calling_line():
 """ Internal. Inspects the current python call stack and returns a nice string description of the line of code that called gnumpy. """
 stack = _pdb.traceback.extract_stack()[::-1] # newest first
 stack = stack[findIndex2(stack, lambda stackElt: stackElt[0] != stack[0][0]):] # skip any gnumpy procs on the stack
 def stackFrameToString(frame): return 'File "%s", line %d, in function %s:    %s' % (frame[0], frame[1], frame[2], ( '<command unknown>' if frame[3]==None else frame[3]))
 ret = stackFrameToString(stack[0])
 for frame in stack[1:]:
  if 'File "<ipython console>",' in ret: break
  if 'File "<stdin>",' in ret: break
  ret += '\n  Called by: ' + stackFrameToString(frame)
 return ret

def memory_allocators(minimum_n_bytes=1):
 """ Prints a list of lines in your code that caused allocated GPU memory that's still in use. """
 if not track_memory_usage:
  print 'The variable gnumpy.track_memory_usage must be set to True, to enable memory data collection (which can slow down your program a lot).'
  return
 for line, (n,amt) in sorted(memoryUsers.items(), key=lambda x:x[1][1]) [::-1] :
  if amt >= minimum_n_bytes:
   print '%d objects, totalling %s, that are still in use, were allocated by: %s' % t3(n, n_bytes_str(amt), line)
   print
 


# ------------------------------------------------------------------------------- external procs

def status():
 if useGpu=='no': print 'gnumpy is running on the CPU, i.e. in simulation mode. The data type is float%s.' % _precision
 if useGpu=='yes':
  if boardId==None: print 'gnumpy is planning to run on a GPU, but hasn\'t yet chosen & initialized a board.'
  else: print 'gnumpy is running on GPU board #%d.' % boardId
 print '%s of gpu memory are in use, of which at least %s can be freed immediately by gnumpy.free_reuse_cache().' % t2(n_bytes_str(memoryInUse), n_bytes_str(__builtin__.sum( size*len(cms)*4 for size, cms in cmsForReuse.items())))
 
 
  
def rand_base(shapeInfo, distribution, zero_d_means_scalar):
 if len(shapeInfo)==1 and isSequence(shapeInfo[0]): zero_d_means_scalar = False; shapeInfo = shapeInfo[0]
 ret = empty(shapeInfo)
 {'uniform': cmType.fill_with_rand, 'normal': cmType.fill_with_randn}[distribution](ret.base)
 if ret.size!=0 and doExpensiveCheck(): assert ret.sum() < 100 + 2*ret.size, 'numerical gpu error: rand() gave a result>100'
 if len(shapeInfo) == 0 and zero_d_means_scalar: return ret.item()
 else: return ret

def tile(a, reps):
 if isNumber(reps): reps = t1(reps)
 reps = tuple(reps) # for generator expressions
 if isNumber(a):
  ret = empty(reps)
  ret.base.assign(a)
  return ret
 a = as_garray(a)
 if len(reps) > a.ndim: a = a.add_axes(len(reps))
 if len(reps) < a.ndim: reps = extend_shape(reps, a.ndim) # now len(reps)==a.ndim
 retShape = tuple([ a.shape[i] * reps[i] for i in ranlen(reps)])
 if prodT(retShape)==0: return zeros(retShape)
 if prodT(reps)==1: return a
 for i in range(a.ndim-1): # merge replication requests on adjacent axes, for efficiency.
  if reps[i]!=1 and reps[i+1]!=1 and a.shape[i]==1: return a.reshape(deleteT2(a.shape, i)).tile(reps[:i]+t1(prodT(reps[i:i+2]))+reps[i+2:]).reshape(map(operator.mul, a.shape, reps))
 def dataIDone(nextA, i): return nextA.reshape(modifyT(a.shape, i, a.shape[i]*reps[i])).tile(modifyT(reps, i, 1))
 if reps[0]!=1: # replicating rows is easy and efficient: just repeat the data a number of times.
  temp = empty(t2(reps[0], a.size)) # shape doesn't matter because dataIDone changes it
  tempCm = temp.base_shaped(1)
  if reps[0]>=1:
   cm_row_slice_read(tempCm, 0, 1).assign(a.base_as_row())
   nCopiesDone = 1
   while nCopiesDone < reps[0]:
    nNow = __builtin__.min(nCopiesDone, reps[0]-nCopiesDone)
    cm_row_slice_read(tempCm, nCopiesDone, nCopiesDone + nNow).assign(cm_row_slice_read(tempCm, 0, nNow))
    nCopiesDone += nNow
  return dataIDone(temp, 0)
 # the general case is repeating a subset (aot the whole array) n times, before moving on to the next subset
 # using a transpose with the right shape, the subsets can become columns. those can be lengthened because that is replicating rows; a second transpose makes them now-lengthened subsets again
 axis = __builtin__.min( i for i in range(a.ndim) if reps[i]!=1)
 return dataIDone(a.reshape_2d(axis).T.tile(t2(reps[axis], 1)).T, axis)
 
def is_garray(x): return isinstance(x, garray)
def is_array(x): return is_garray(x) or isNpArray(x)

def rand(*shapeInfo):
 """ the desired array shape can be entered either as integers or as a tuple of integers. If you enter a tuple you always get an array; if you enter nothing you get a scalar. """
 return rand_base(shapeInfo, 'uniform', True)

def randn(*shapeInfo):
 """ the desired array shape can be entered either as integers or as a tuple of integers. If you enter a tuple you always get an array; if you enter nothing you get a scalar. """
 return rand_base(shapeInfo, 'normal', True)

def empty(shape):
 if isSequence(shape) or isGenerator(shape): shape = tuple(shape)
 else: shape = t1(shape)
 return garray(new_cm(prodT(shape)), shape, None)

def zeros (shape): return tile(0, shape)
def ones (shape): return tile(1, shape)

def seed_rand(seed=None):
 init_gpu()
 if seed==None: seed = int(_time.time())
 _cudamat.CUDAMatrix.init_random(seed)

def dot(a1, a2):
 # internally: for matrix-matrix multiplies only; vectors are treated like special cases.
 a1 = as_garray(a1); a2 = as_garray(a2)
 if a1.ndim==0 or a2.ndim==0: return a1*a2
 if a1.ndim==a2.ndim==1:
  if a1 is a2: return sum(a1**2)
  else: return dot(a1.reshape(1, a1.size), a2.reshape(a2.size, 1)).item()
 if a1.ndim==2 and a2.ndim==1: return dot(a1, a2.reshape(a2.size, 1)).ravel() # treat a2 like a column vector
 if a1.ndim==1 and a2.ndim==2: return dot(a1.add_axes(2), a2)[0]   # treat a1 like a row vector
 if a1.shape[-1] != a2.shape[-2]: raise ValueError('arrays not aligned for dot product. a dot product was requested of arrays with shapes %s and %s' % t2(a1.shape, a2.shape))
 if a1.ndim==a2.ndim==2:
  retShape = t2(a1.shape[0], a2.shape[1])
  if a1.shape[1]==0: return zeros(retShape) # cudamat bug workaround
  ret = empty(retShape)
  if ret.size!=0: _cudamat.dot(a2.base_as_2d(), a1.base_as_2d(), ret.base_as_2d())
  return ret
 if a1.ndim >= 2 and a2.ndim >= 2:
  # this is not necessarily fast, because if a2.ndim>=3 then it involves a transpose
  a12 = ( a1.reshape_2d(-1) if a1.ndim!=2 else a1)
  a22 = ( a2.transpose(t1(a2.ndim-2) + rangeT(a2.ndim-2) + t1(a2.ndim-1)).reshape_2d(1)
          if a2.ndim!=2 else
          a2)
  retShape = deleteT2(a1.shape, -1) + deleteT2(a2.shape, -2)
  return dot(a12, a22).reshape(retShape)
 raise NotImplementedError('dot with arguments of shapes %s and %s' % t2(a1.shape, a2.shape))

def outer(vec1, vec2): return dot(vec1.ravel()[:, newaxis], vec2.ravel()[newaxis, :])

def concatenate(arrays, axis=0):
 arrays = mapTSimplified(arrays, as_garray)
 if axis<0: axis += arrays[0].ndim
 if not isSequence(arrays) or not isNumber(axis): raise ValueError('wrong argument types to gnumpy.concatenate: expected <arrays> to be a sequence and <axis> to be a number, but got types %s and %s.' % t2(type(arrays), type(axis)))
 if axis not in range(arrays[0].ndim): raise ValueError('bad axis number (%d) specified (the first array has %d axes)' % t2(axis, arrays[0].ndim))
 if not allTheSame( deleteT2(a.shape, axis) for a in arrays): raise ValueError('array dimensions must agree except possibly for axis #%d. The given array shapes are: %s' % t2(axis, tuple( a.shape for a in arrays)))
 finalShape = modifyT(arrays[0].shape, axis, __builtin__.sum( a.shape[axis] for a in arrays))
 if axis==0:
  ret = empty(finalShape)
  nextI = 0
  for a in arrays:
   cm_row_slice_read(ret.base_shaped(ret.ndim), nextI, nextI+a.size).assign(a.base_shaped(a.ndim))
   nextI += a.size
  return ret
 else:
  return concatenate(tuple([ a.reshape_2d(axis).T for a in arrays]), 0).T.reshape(finalShape)
 
def where(a, *vararg):
 """
 Note: if only one argument is provided, the returned value will be a tuple of *numpy* arrays of integer indices (gpu arrays can only contain floats).
 """
 if vararg==t0: return numpy.where(as_numpy_array(a))
 assert len(vararg)==2, 'wrong number of arguments to gnumpy.where()'
 return garray(numpy.where(as_numpy_array(a), as_numpy_array(vararg[0]), as_numpy_array(vararg[1])))

def nonzero(a):
 """ See notes for where(). """
 return where(a)
 
newaxis = None

def eye(n): return diagflat(ones(n))

def diagflat(a, k=0):
 if is_garray(a): return a.diagflat(k)
 else: return numpy.diagflat(a,k)

def tensordot(a, b, axes=2):
 if isNumber(axes): return dot(a.reshape_2d(a.ndim-axes), b.reshape_2d(axes)).reshape(a.shape[:a.ndim-axes] + b.shape[axes:])
 assert len(axes)==2 and len(axes[0])==len(axes[1]), 'the axes parameter to gnumpy.tensordot looks bad'
 aRemove, bRemove = t2(tuple(axes[0]), tuple(axes[1]))
 return tensordot(a.transpose(filter(lambda x: x not in aRemove, rangeT(a.ndim)) + aRemove),
                  b.transpose(bRemove + filter(lambda x: x not in bRemove, rangeT(b.ndim))),
                  len(aRemove))

 
 
# ------------------------------------------------------------------------------- reductors

def reductor_base(x, axis, gpuOp, npOp):
 if isTijmen: numTimeIncurred(x.size, '%s onDim0=%s' % t2(npOp.__name__, axis in t2(0, None)))
 if isNpArray(x): return npOp(x, axis)
 if not is_garray(x): x = garray(x)
 if gpuOp==None: return garray(npOp(x.as_numpy_array(), axis))
 else: return gpuOp(x, axis)

def all(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, garray.all, numpy.all)

def any(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, garray.any, numpy.any)

def sum(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, garray.sum, numpy.sum)

def mean(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, garray.mean, numpy.mean)

def max(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, garray.max, numpy.max)

def min(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, garray.min, numpy.min)

def prod(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, None, numpy.prod)

def std(x, axis=None):
 """ On numpy arrays this returns a numpy array; on garrays and other array-likes this returns a garray. """
 return reductor_base(x, axis, None, numpy.std)



# ------------------------------------------------------------------------------- elementwise operations

def elementwise_base(x, opGpu, opNp):
 if isNumber(x): return check_number_types(float(opNp(x)))
 if opGpu==None or isNpArray(x): # else, time admin happens in the method
  if isTijmen: numTimeIncurred(x.size, opNp.__name__)
 if is_garray(x):
  if opGpu==None: return check_number_types(garray(opNp(x.as_numpy_array())))
  else: return check_number_types(opGpu(x))
 if isNpArray(x):
  if x.ndim==0: return check_number_types(numpy.array(opNp(x)))
  else: return check_number_types(opNp(x))
 raise TypeError('value %s of unexpected type %s provided to %s()' % t3(x, type(x), str(opNp).split("'")[1]))

def abs(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.abs, numpy.abs)

def exp(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.exp, numpy.exp)

def isinf(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.isinf, numpy.isinf)

def isnan(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.isnan, numpy.isnan)

def log(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.log, numpy.log)

def log_1_plus_exp(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.log_1_plus_exp, lambda x: 1.+exp(x))
 
def logistic(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.logistic, lambda x: 1./(1. + exp(-x)))
 
def negative(x):
 """
 Like -x, except that a zero dimensional numpy array input results in a numpy array return value.
 This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats).
 """
 return elementwise_base(x, operator.neg, operator.neg)

def sign(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.sign, numpy.sign)

def sqrt(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.sqrt, numpy.sqrt)

def tanh(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, garray.tanh, numpy.tanh)
 
def log10(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, None, numpy.log10)
 
def log2(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, None, numpy.log2)
 
def cos(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, None, numpy.cos)
 
def sin(x):
 """ This works on garrays, numpy arrays, and numbers, preserving type (though all numbers become floats). """
 return elementwise_base(x, None, numpy.sin)
 

 
 

class garray(object):
 """
 A class designed to interface like numpy arrays, and internally do its work on a GPU.
 Documentation can be found at http://www.cs.toronto.edu/~tijmen/gnumpy.html
 """

 # ------------------------------------------------------------------------------- internal aux

 def set_shape_info(self, shape): # setting these as attributes rather than properties saves exec time
  self.shape = shape
  self.size = prodT(shape)
  self.ndim = len(shape)
 
 @property
 def nbytes(self): return self.size * 4
 @property
 def nMBytes(self): return self.nbytes / 2**20
  
 def base_shaped(self, nDimsAsRows): return cm_reshape(self.base, t2(prodT(self.shape[:nDimsAsRows]), prodT(self.shape[nDimsAsRows:])))
 def base_as_row(self): return cm_reshape(self.base, t2(1, self.size))
 def base_as_2d(self): return self.base.reshape(t2(self.shape[1], self.shape[0])) # optimized from self.base_shaped(1) by inlining
 
 def new_cm(self, nDimsAsRows=0): return new_cm(t2(prodT(self.shape[:nDimsAsRows]), prodT(self.shape[nDimsAsRows:]))) # same size as self, with given shape
 
 def new(self, cm): return garray(cm, self.shape, None) # short notation for the result of elementwise ops
 
 def tile_to_broadcast(self, otherShape, indicesToBroadcast='all'):
  """ self.shape and otherShape must already be of the same length. otherShape is relevant only where self.shape is 1. """
  if otherShape == self.shape: return self
  assert self.ndim == len(otherShape), 'dimensionality mismatch in tile_to_broadcast'
  if indicesToBroadcast=='all': indicesToBroadcast = tuple( i for i in range(self.ndim) if self.shape[i]==1 and otherShape[i]!=1)
  return self.tile( ( 1 if i not in indicesToBroadcast else otherShape[i] ) for i in range(self.ndim))
 
 def broadcastable_op(self, other, operatorName):
  """
  accepted ops: "add", "multiply", "less than", "greater than", "pow".
  other must be either scalar or garray.
  """
  basicHandler = {'add': cmType.add, 'multiply': cmType.mult, 'less than': cmType.less_than, 'greater than': cmType.greater_than, 'pow': _cudamat.pow}[operatorName]
  if (isNumber(other) or (other.size==1 and other.ndim <= self.ndim)): # having other be a scalar is faster than doing a broadcast
   if isTijmen: numTimeIncurred(self.size, 'AS eltwise')
   return self.new(basicHandler(self.base_as_row(), ( other.item() if is_garray(other) else other), self.new_cm()))
  if operatorName=='pow': raise NotImplementedError('a**b where b is anything other than a scalar')
  other = as_garray(other)
  if self.ndim > other.ndim: other = other.add_axes(self.ndim)
  if self.ndim < other.ndim: return self.add_axes(other.ndim).broadcastable_op(other, operatorName)
  if operatorName in t2('less than', 'greater than'):
   self2 = self.tile_to_broadcast(other.shape)
   if isTijmen: numTimeIncurred(self.size, 'eltwise binary, no bc')
   return self2.new(basicHandler(self2.base_as_row(), other.tile_to_broadcast(self2.shape).base_as_row(), self2.new_cm()))
  if self.ndim < other.ndim: return other.broadcastable_op(self, operatorName) # now self.ndim == other.ndim
  selfToBroadcast =  tuple( self.shape[i]==1 and other.shape[i]!=1 for i in range(self.ndim))
  otherToBroadcast = tuple( other.shape[i]==1 and self.shape[i]!=1 for i in range(self.ndim))
  bc = otherToBroadcast; bci = argSelect(bc, echo)
  if anyT(selfToBroadcast) and anyT(otherToBroadcast): return self.broadcastable_op(other.tile_to_broadcast(self.shape, bci), operatorName)
  if anyT(selfToBroadcast): return other.broadcastable_op(self, operatorName) # now only other may have dims that need to be broadcast
  if anyT( other.shape[i] not in t2(1, self.shape[i]) for i in range(self.ndim)): raise ValueError('shape mismatch: objects cannot be broadcast to a single shape')
  if not anyT(otherToBroadcast): # handle case: nothing to bc
   if isTijmen: numTimeIncurred(self.size, 'eltwise binary, no bc')
   return self.new(( cmType.add if operatorName=='add' else cmType.mult)(self.base_as_row(), other.base_as_row(), self.new_cm()))
  if self.size==0: return self
  if bci == ranlen(bci): # handle case: only the first dims need broadcasting
   if operatorName in ('multiply', 'add') and isTijmen: # using optimized cuda code
    ret = empty(self.shape)
    axis0len = prodT(self.shape[:len(bci)])
    axis1len = prodT(self.shape[len(bci):])
    nThreadsPerBlock = 512
    nBlocks = axis1len//nThreadsPerBlock+1
    cudaFn = getattr(_cudamat._cudamat, '%sBcAxis0' % operatorName)
    cudaFn.restype = _ctypes.c_int
    assert 0==cudaFn(ctInt(nBlocks), ctInt(nThreadsPerBlock), self.base.p_mat, other.base.p_mat, ret.base.p_mat, ctInt(axis0len), ctInt(axis1len))
    if isTijmen: numTimeIncurred(self.size, 'eltwise bc axis 0')
    return ret
   #return self.new(( cmType.add_col_vec if operatorName=='add' else cmType.mult_by_col)(self.base_shaped(len(bci)), other.base_as_row(), self.new_cm(len(bci))))
  if bci == rangeT(self.ndim-len(bci), self.ndim): # handle case: only the last dims need broadcasting
   if isTijmen: numTimeIncurred(self.size, 'eltwise bc axis -1')
   return self.new(( cmType.add_row_vec if operatorName=='add' else cmType.mult_by_row)(self.base_shaped(self.ndim-len(bci)), other.base_shaped(self.ndim-len(bci)), self.new_cm(self.ndim-len(bci))))
  # remaining case: broadcasting neither just the first dims nor just the last dims. this can be done very intelligently, but for now I won't bother
  if operatorName=='multiply' and len(bci)==1 and cudamatHas('multiplyBcAxis1'): # special case: using optimized multiplyBcAxis1 (my cuda code)
   ret = empty(self.shape)
   axisI = bci[0]
   axis0len = prodT(self.shape[:bci[0]])
   axis1len = self.shape[bci[0]]
   axis2len = prodT(self.shape[bci[0]+1:])
   _cudamat._cudamat.multiplyBcAxis1.restype = _ctypes.c_int
   assert 0==_cudamat._cudamat.multiplyBcAxis1(ctInt(__builtin__.min(512, axis2len)),
                                          self.base.p_mat,
                                          other.base.p_mat,
                                          ret.base.p_mat,
                                          ctInt(axis0len),
                                          ctInt(axis1len),
                                          ctInt(axis2len), 
                                          )
   if isTijmen: numTimeIncurred(self.size, 'eltwise bc axis 1')
   return ret
  return self.broadcastable_op(other.tile_to_broadcast(self.shape, bci[:1]), operatorName)

 def elementwise_unary(self, handler):
  if isTijmen: numTimeIncurred(self.size, handler.__name__)
  return check_number_types(self.new(handler(self.base_as_row(), self.new_cm())))

 def reduction_base(self, operatorName, axis):
  if axis==None: return self.ravel().reduction_base(operatorName, 0).item()
  if not isNumber(axis): raise TypeError('the value %s is not appropriate for the "axis" parameter.' % str(axis))
  if axis < -self.ndim or axis>=self.ndim: raise ValueError('axis (%d) out of bounds for an array with %d axes.' % t2(axis, self.ndim))
  axis = int(axis) % self.ndim
  if self.size==0:
   retShape = deleteT2(self.shape, axis)
   if operatorName=='sum': return zeros(retShape)
   elif operatorName=='max': return tile(-inf, retShape)
   else: assert False
  if operatorName=='max' and axis==0 and cudamatHas('maxAxis0'): # my own fast implementation
   ret = empty(self.shape[1:])
   ctInt = _cudamat.ct.c_int
   nThreadsPerBlock = 32
   gridX, gridY = ((ret.size+nThreadsPerBlock-1)//nThreadsPerBlock), 1
   while gridX>65535: gridY*=2; gridX = (gridX+1)//2;
   _cudamat._cudamat.maxAxis0.restype = _ctypes.c_int
   assert 0==_cudamat._cudamat.maxAxis0(ctInt(gridX), ctInt(gridY), ctInt(nThreadsPerBlock), self.base.p_mat, ret.base.p_mat, ctInt(self.shape[0]), ctInt(ret.size))
   return ret
  if axis==0 and operatorName=='max': # max over rows is not yet supported in cudamat
   return self.reshape_2d(1).T.max(1).reshape(self.shape[1:])
  if axis==0 and self.ndim==1 and self.size>5000 and operatorName=='sum': # optimization. apparently, cudamat is not maximally efficient.
   n = int(numpy.sqrt(self.size-1))
   return self[:n*n].reshape(t2(n, n)).reduction_base(operatorName, 0).reduction_base(operatorName, 0) + self[n*n:].reduction_base(operatorName, 0)
  if operatorName=='sum':
   chunkSize = 1024*256 # sum over longer dimensions fails in cudamat
   nChunks = (self.shape[axis] + chunkSize-1) // chunkSize
   if nChunks>1:
    return reduceAdd( self[t1(slice(None)) * axis + t1(slice(chunkI*chunkSize, __builtin__.min(self.shape[axis], (chunkI+1)*chunkSize)))].reduction_base(operatorName, axis)
                      for chunkI in range(nChunks))
  if operatorName=='max' and self.isnan().any2(): # cudamat bug workaround
   return garray(self.asarray().max(axis))
  operatorInCm = {'sum': cmType.sum, 'max': cmType.max}[operatorName]
  if axis==0: return check_number_types(garray(operatorInCm(self.base_shaped(1), 1, new_cm(prodT(self.shape[1:]))), self.shape[1:], None))
  if axis==self.ndim-1:
   if self.ndim!=2: return self.reshape_2d(-1).reduction_base(operatorName, 1).reshape(self.shape[:-1])
   if self.ndim==2:
    chunkSize = 2**16-1
    nChunks = (len(self) + chunkSize-1) // chunkSize
    if nChunks>1: # cudamat chokes on big arrays, so break it in pieces for cudamat
     chunks = tuple([ self[chunkI*chunkSize : __builtin__.min((chunkI+1)*chunkSize, len(self))]
                      for chunkI in range(nChunks)])
     return concatenate([ chunk.reduction_base(operatorName, 1) for chunk in chunks])
    else: # small array
     return check_number_types(garray(operatorInCm(self.base_shaped(1), 0, new_cm(t2(len(self), 1))), t1(len(self)), None))
  return self.transpose_simple(axis).reduction_base(operatorName, 0).transpose_simple(-axis)
 

 
 # ------------------------------------------------------------------------------- external misc non-numerical
 
 def __init__(self, data, copy=True, ndmin=0):
  """ the parameters mean the same as in numpy.array() """
  if type(data)!=cmType: assert copy in t2(True, False) and isNumber(ndmin), 'garray() parameters copy=%s, ndmin=%s are not of the right type' % t2(str(copy), str(ndmin))
  if type(data)==cmType: # internal use only. the 3 arguments are, unlike their names suggest, the .base, .shape, .is_alias_of
   self.base = data
   self.set_shape_info(copy)
   self.is_alias_of = ndmin
   if self.is_alias_of==None and track_memory_usage:
    self.allocating_line = calling_line()
    memoryUsers[self.allocating_line] = t2(memoryUsers[self.allocating_line][0]+1, memoryUsers[self.allocating_line][1]+self.size*4)
  elif is_garray(data):
   if ndmin>0: data = data.add_axes(ndmin)
   garray.__init__(self, 
    ( new_cm(data.size).assign(data.base_as_row()) if copy else data.base),
    data.shape,
    ( None if copy else data))
  elif isGenerator(data): garray.__init__(self, tuple(data), ndmin=ndmin)
  elif isSequence(data):
   if len(data)==0 or not any2_(data, is_garray): garray.__init__(self, numpy.array(data, ndmin=ndmin), copy=False)
   else: garray.__init__(self, concatenate( as_garray(element)[None] for element in data), ndmin=ndmin) # no need to copy, because concat copies.
  else: # remaining cases. essentially init from numpy array.
   npa = numpy.array(data, copy=False) # in case data was a number
   if str(npa.dtype) in t2('object', '|S3'): raise TypeError('Cannot convert "%s" to a garray.' % data) 
   # we're not using the cudamat constructor, because that always allocs gpu mem, and this way the mem may come from re-use.
   cm = new_cm(npa.size)
   if not hasattr(cm, 'numpy_array'):
    #cm.copy_to_host() # if cm was created using cudamat.empty, this is needed to associate cm with a numpy array
    # follows an inlined version of the relevant portion of cm.copy_to_host(). This is quicker because it doesn't actually copy.
    cm.numpy_array = numpy.empty((cm.mat.size[0], cm.mat.size[1]), dtype=numpy.float32, order='F')
    cm.mat.data_host = cm.numpy_array.ctypes.data_as(_ctypes.POINTER(_ctypes.c_float))
    cm.mat.on_host = 1
   if npa.size!=0: cm.numpy_array[:] = npa.reshape(t2(-1, 1), order='C') # no cudamat.reformat is needed, because that's only dtype and order change, which are handled by the assignment anyway
   cm.copy_to_device()
   garray.__init__(self, cm, extend_shape(npa.shape, ndmin), None)

 def __new__(cls, *args, **kwarg): return object.__new__(cls)
   
 def as_numpy_array(self, dtype=numpy.float64):
  if self.size==0: return numpy.zeros(self.shape, dtype)
  return numpy.array(self.base_as_row().asarray(), copy=True, order='C', dtype=dtype).reshape(self.shape)
 
 asarray = as_numpy_array # the cudamat name
 
 def astype(self, type): return self.asarray().astype(type)
 
 tile = tile
 
 def ravel(self): return self.reshape(-1)
 
 def item(self): return self.as_numpy_array().item()
 
 def add_axes(self, finalNdim): return self.reshape(extend_shape(self.shape, finalNdim))

 def sort(self, axis=-1, kind='quicksort', order=None):
  """ like numpy.sort, this sorts in place and returns None. """
  temp = self.as_numpy_array()
  temp.sort(axis, kind, order)
  self[:] = temp
 
 def reshape(self, *newShape):
  if len(newShape)==1 and not isNumber(newShape[0]): newShape = tuple(newShape[0])
  if not all2_(newShape, isNumber): raise TypeError('the parameters to reshape don\'t look like a valid shape')
  if -1 in newShape:
   if prodT(newShape)==0: raise ValueError("-1 as a parameter to reshape is not allowed if one of the other parameters is zero.")
   newShape = modifyT(newShape, operator.indexOf(newShape, -1), self.size//-prodT(newShape))
  if prodT(newShape) != self.size: raise ValueError('the total number of items cannot be changed in a reshape')
  return garray(self.base, newShape, self)
 
 def reshape_2d(self, n_dimensions_as_rows):
  """ reshapes to 2 axes. The first <n_dimensions_as_rows> axes of the array become the first axis of the returned value. The remaining ones form the second axis. """
  if n_dimensions_as_rows<0: n_dimensions_as_rows += self.ndim
  return self.reshape(t2(prodT(self.shape[:n_dimensions_as_rows]), prodT(self.shape[n_dimensions_as_rows:])))
 
 @property
 def T(self):
  if self.ndim==2: # base case
   if self.size==0: return self.reshape(reverseT(self.shape)) # cudamat bug workaround
   if self.shape[1]>1e6: # cudamat bug workaround. with 2m columns it fails
    return concatenate([ self[:, i*10**6 : (i+1)*10**6].T for i in range((self.shape[1]+10**6-1)//10**6)])
   if self.shape[0]>1e6: # cudamat bug workaround. using concat is not an option, because that uses transpose.
    ret = empty(reverseT(self.shape))
    for i in range((self.shape[0]+10**6-1)//10**6):
     ret[:, i*10**6 : (i+1)*10**6] = self[i*10**6 : (i+1)*10**6].T 
    return ret
   return garray(self.base_as_2d().transpose(new_cm(reverseT(self.shape))), reverseT(self.shape), None)
  else: return self.transpose()  

 def transpose_simple(self, nDimsToGroup):
  """ shifts the first <nDimsToGroup> axes to the end, and the remaining ones to the start. This returns a new array, not an alias. """
  if nDimsToGroup<0: nDimsToGroup += self.ndim
  return self.reshape_2d(nDimsToGroup).T.reshape(self.shape[nDimsToGroup:] + self.shape[:nDimsToGroup])
 
 def transpose(self, *axes):
  """ like numpy.transpose, except that this doesn't return an alias, but rather a new array. """
  # This is not really supported by cudamat, so it takes creativity. I handle a variety of cases differently.
  if len(axes)==1 and not isNumber(axes[0]): axes = tuple(axes[0])
  if axes==t0: axes = reverseT(rangeT(self.ndim))
  if axes == rangeT(self.ndim): return self.copy()
  if tuple(sorted(axes)) != rangeT(self.ndim): raise ValueError("%s is not a valid argument to transpose() of an array of %d axes" % t2(axes, self.ndim))
  for i in range(self.ndim-1): 
   if axes[i+1]==axes[i]+1: return (self. # see if the task can be simplified by collapsing some axes that are kept adjacent
    reshape(self.shape[:axes[i]] + t1(prodT(self.shape[axes[i]:axes[i]+2])) + self.shape[axes[i]+2:]).
    transpose((originalAxisI-(originalAxisI>axes[i])) for originalAxisI in deleteT2(axes, i+1)).
    reshape(self.shape[axisI] for axisI in axes))
  if self.ndim==3 and hasattr(_cudamat, '_cudamat') and cudamatHas('transpose3') and self.size!=0:
   reorderingI = {(0, 2, 1): 0, (1, 0, 2): 1, (2, 1, 0): 2}[axes]
   ret = empty(tuple( self.shape[axisI] for axisI in axes))
   gridX, gridY = (self.size+511)//512, 1
   while gridX>65535: gridY*=2; gridX = (gridX+1)//2;
   _cudamat._cudamat.transpose3.restype = _ctypes.c_int
   assert 0==_cudamat._cudamat.transpose3(ctInt(gridX), ctInt(gridY), self.base.p_mat, ret.base.p_mat, ctInt(self.shape[0]), ctInt(self.shape[1]), ctInt(self.shape[2]), ctInt(reorderingI))
   return ret
  def shiftAxesRight(shiftN): return self.transpose_simple(-shiftN).transpose( (axisI+shiftN)%self.ndim for axisI in axes)
  for i in range(self.ndim-1): # see if the task can be simplified by rotating axes right by 1. if so, the loop before this one can simplify further
   if axes[i:i+2] == t2(self.ndim-1, 0): return shiftAxesRight(1)
  # no further simplifications can be done. we need to proceed with a loop over the first axis. First rotate the intended axis to position 0.
  if axes[0]!=0: return shiftAxesRight(-axes[0])
  ret = empty( self.shape[axisI] for axisI in axes)
  for i in range(self.shape[0]): ret[i] = self[i].transpose( x-1 for x in axes[1:])
  return ret
   
 def copy(self): return garray(self, copy=True)
 
 def diagflat(self, k=0):
  if self.ndim!=1: return self.ravel().diagflat(k)
  if k!=0: raise NotImplementedError('k!=0 for garray.diagflat')
  selfSize = self.size
  ret = zeros(t2(selfSize, selfSize))
  ret.ravel()[:-1].reshape(t2(selfSize-1, selfSize+1))[:, 0] = self[:-1]
  if selfSize!=0: ret.ravel()[-1] = self[-1]
  return ret
   
 def diagonal(self):
  if self.ndim==1: return self.diagflat()
  if self.ndim==2:
   if self.shape[0] > self.shape[1]: return self[:self.shape[1]].diagonal()
   if self.shape[1] > self.shape[0]: return self[:, :self.shape[0]].diagonal()
   return self.ravel()[::self.shape[0]+1]
  raise NotImplementedError('garray.diagonal for arrays with ndim other than 1 or 2.')
 def diag(self): return self.diagonal()
  


 # ------------------------------------------------------------------------------- elementwise type checking
 
 def all_real(self):
  """ returns True iff all array elements are regular floats, as opposed to inf's, -inf's, and NaN's.  """
  return (self*0).sum()==0
  
 def isinf(self):
  """ elementwise, checking for inf or -inf. """
  return 1 - self.isreal() - self.isnan()
 
 def isreal(self):
  """ elementwise, checking for real numbers. See also .all_real() """
  return (self<numpy.inf) * (self>-numpy.inf)
 
 def isnan(self): 
  """ elementwise, checking for NaN's. """
  return (self>0) + (self<1) < .5

 def isnumber(self):
  """ elementwise, checking for anything other than NaN's """
  return (self>0) + (self<1) > .5
 
 
 
 # ------------------------------------------------------------------------------- external misc numerical
 
 def __abs__(self): return self.elementwise_unary(_cudamat.abs)
 def abs(self): return __builtin__.abs(self)
 def as_bool(self): return self!=0
 def exp(self): return self.elementwise_unary(_cudamat.exp)
 def log(self): return self.elementwise_unary(_cudamat.log)
 def log_1_plus_exp(self): return self.elementwise_unary(_cudamat.log_1_plus_exp)
 def logistic(self): return self.elementwise_unary(_cudamat.sigmoid)
 sigmoid = logistic
 def sign(self): return self.elementwise_unary(cmType.sign)
 def sqrt(self): return self.elementwise_unary(_cudamat.sqrt)
 def tanh(self): return self.elementwise_unary(_cudamat.tanh)
 

 def sum(self, axis=None): return self.reduction_base('sum', axis)
 def max(self, axis=None): return self.reduction_base('max', axis)
 def mean(self, axis=None): return self.sum(axis) / ( self.size if axis==None else self.shape[axis])
 def argmax(self, axis=None): return numpy.argmax(self.asarray(), axis)
 def argmin(self, axis=None): return numpy.argmin(self.asarray(), axis)
 def min(self, axis=None): return -(-self).max(axis)
 def all(self, axis=None): return ( True if self.size==0 else (self.as_bool()).min())
 def any(self, axis=None): return ( False if self.size==0 else (self.as_bool()).max())
 
 def all2(self, axis=None): return 1-(1-self).any2(axis)  # optimized for when I'm sure that the content is boolean
 def any2(self, axis=None): return self.sum(axis) > 0  # optimized for when I'm sure that the content is boolean
 
 def rand(self, distribution = 'uniform'):
  """
  returns a new garray, of the same shape as self, filled with random numbers.
  <distribution> can be either 'uniform' or 'normal'.
  """
  return rand_base(self.shape, distribution, False)

 def euclid_norm(self): return self.base.euclid_norm()

 dot = dot
 where = where
 nonzero = nonzero
 
 def __nonzero__(self): return self.size==1 and self.item()!=0
 
 
 # ------------------------------------------------------------------------------- operator overloads, numerical
 
 def __add__(self, other): return check_number_types(self.broadcastable_op(as_garray_or_scalar(other), 'add'))
 def __mul__(self, other): return check_number_types(self.broadcastable_op(as_garray_or_scalar(other), 'multiply'))
 def __or__(self, other): return (self.as_bool() + other.as_bool()).as_bool()
 def __and__(self, other): return self.as_bool() * other.as_bool()
 
 def __pow__(self, other, modulo=None):
  if modulo!=None: raise NotImplementedError('power with modulo')
  if isNumber(other) and other==2: return self*self # faster
  return self.broadcastable_op(as_garray_or_scalar(other), 'pow')
 
 
 # the following would be a lot simpler if I wouldn't have to deal with nans
 
 def __lt__(self, other): return check_number_types(self.broadcastable_op(as_garray_or_scalar(other), 'less than'))
 
 def __gt__(self, other): return check_number_types(self.broadcastable_op(as_garray_or_scalar(other), 'greater than'))
 
 def __le__(self, other): return self.isnumber() * as_garray(other).isnumber() * (1-(self>other))
 
 def __ge__(self, other): return self.isnumber() * as_garray(other).isnumber() * (1-(self<other))
 
 def __ne__(self, other): return ( 1-(self==other) if type(other) in castableTypes else True)
 
 def __eq__(self, other): return ( (self<=other) * (self>=other) if type(other) in castableTypes else False)
 
 def eq2(self, other):
  """
  Returns a boolean: True if self and other are the same (arrays with the same shape and contents); False otherwise.
  This is what == does on most Python objects (on arrays it's been strangely overloaded though).
  garrays compare equal to numpy arrays with the same contents, even if the data types differ.
  """
  if self is other: return True
  if not is_array(other): return False
  if self.shape != other.shape: return False
  return all(self==other)==1
 
 def __sub__(self, other):
  if is_garray(other) and other.shape==self.shape: # use specialized method
   return self.new(self.base_as_row().subtract(other.base_as_row(), self.new_cm()))
  else: return self + -as_garray(other) # if i need to broadcast, making use of the row add and col add methods is probably faster
 
 def __div__(self, other):
  if isNumber(other): return self * (1./other)
  other = as_garray(other)
  return self * other.new(other.base_as_row().reciprocal(other.new_cm()))

 def __rmul__(self, other): return self*other
 def __radd__(self, other): return self+other
 def __rsub__(self, other): return other + -self
 def __rdiv__(self, other): return as_garray(other) / self
 def __rpow__(self, other): raise NotImplementedError('a**b where only b is a garray')
 
 def __pos__(self): return self
 def __neg__(self): return self*-1
 
 def __iadd__(self, other): self[t0] = self+other; return self # not as direct as it might have been, but the effect is the same. "self[:]" doesn't work for 0das.
 def __imul__(self, other): self[t0] = self*other; return self
 def __isub__(self, other): self[t0] = self-other; return self
 def __idiv__(self, other): self[t0] = self/other; return self
 def __imod__(self, other): self[t0] = self%other; return self
 def __ipow__(self, other, modulo=None): self[t0] = self.__pow__(other, modulo); return self


 
 # ------------------------------------------------------------------------------- operator overloads, non-numerical
 
 def __len__(self):
  if self.ndim==0: raise TypeError('len() of unsized object')
  return self.shape[0]
 
 def __getitem__(self, selectors):
  selectors = nonSeqAsS(selectors)
  for i,sel in enumerate(selectors): # deal with newaxis and ellipsis
   if sel is Ellipsis: return self[selectors[:i] + t1(slice(None))* (self.ndim - (countT( x != None for x in selectors)-1)) + selectors[i+1:]] # sel==Ellipsis is bad when sel is an array
   if sel is newaxis: return self.reshape(insertT(self.shape, i, t1(1)))[modifyT(selectors, i, slice(None))]
  if len(selectors) > self.ndim: raise IndexError('more indices than axes')
  if all2_(selectors, isFullSlice): return self
  if allT( isSequence(sel) or is_array(sel) for sel in selectors) and len(selectors)>=2:
   selectors = mapTSimplified(selectors, as_garray)
   if anyT( (sel < 0).sum() > 0 for sel in selectors): raise NotImplementedError('negative indices in index arrays, combined with having multiple indices arrays')
   # ravel the first two dimensions into one, and translate the corresponding indices arrays into one accordingly
   return self.reshape(t1(self.shape[0]*self.shape[1]) + self.shape[2:])[t1(selectors[0]*self.shape[1]+selectors[1]) + selectors[2:]]
  if countT( isSequence(sel) or is_array(sel) for sel in selectors)>1:
   raise NotImplementedError('slicing with more than one sequence/array among the indices, with also other kinds of values among the indices')
  # handle the operations on different axes one by one; earlier axes are handled earlier
  axisI = findIndex2(selectors, lambda i: not isFullSlice(i))
  axisLen = self.shape[axisI]
  axisSelector = selectors[axisI]
  if not all2_(selectors[axisI+1:], isFullSlice): return self[selectors[:axisI+1]][t1(slice(None))*(axisI+(not isNumber(axisSelector))) + selectors[axisI+1:]] # first select on axisI only; then do the further axes.
  # from here, axisI is the only axis on which we don't take a full slice
  if isSlice(axisSelector) and axisSelector.step not in t2(1, None): axisSelector = numpy.arange(axisLen)[axisSelector]
  if isNumber(axisSelector): # selecting a single location on axisI, and thus reducing the dimensionality by 1
   ret = self[selectors[:axisI] + t1(short_slice(read_single_index(axisSelector, axisLen)))]  .reshape(deleteT2(self.shape, axisI))
   return ( ret.item() if ret.shape==t0 else ret) # exception, to have the same behavior as numpy
  if isSequence(axisSelector) or isNpArray(axisSelector): axisSelector = garray(axisSelector)
  if is_garray(axisSelector):
   # a 1d index means re-arranging this axis. I.e. a number of length 1 selections on this axis, concatenated on this axis.
   # other dimensionality means using the raveled version, and then reshaping to reflect the selector dimensionality
   if hasattr(cmType, 'select_columns'):
    if axisI==0:
     if doExpensiveCheck() and (axisSelector> len(self)-.01).sum() !=0: raise IndexError('index %d (found in an indices array) is too large, for an axis of length %d' % t2(max(axisSelector), len(self)))
     if doExpensiveCheck() and (axisSelector<-len(self)-.5).sum() !=0: raise IndexError('index %d (found in an indices array) is too small, for an axis of length %d' % t2(min(axisSelector), len(self)))
     return garray(self.base_shaped(1).select_columns(axisSelector.base_shaped(axisSelector.ndim), new_cm(t2(axisSelector.size, self.size/self.shape[0]))), axisSelector.shape + self.shape[1:], None)
    else: return self.transpose_simple(axisI)[axisSelector].transpose_simple(-axisI)
   else: return (concatenate(tuple( self[modifyT(selectors, axisI, slice(choiceOnThisAxis, choiceOnThisAxis+1))] for choiceOnThisAxis in axisSelector.ravel()), axisI)
                 .reshape(self.shape[:axisI] + axisSelector.shape + self.shape[axisI+1:]))
  if not isSlice(axisSelector): raise ValueError('index not understood: %s' % axisSelector)
  # from here, selector is a simple slice
  sFrom, sTo, sLen = read_simple_slice(axisSelector, axisLen)
  retShape = modifyT(self.shape, axisI, sLen)
  if prodT(retShape)==0: return zeros(retShape)
  if axisI==0: return garray(cm_row_slice_read(self.base_shaped(1), sFrom, sTo), retShape, self) # slice on axis 0 is free, using cm_row_slice_read
  if axisI!=1: return self.reshape(t1(prodT(self.shape[:axisI])) + self.shape[axisI:])[:, sFrom:sTo].reshape(retShape) # redirect: collapse earlier axes into one
  if self.ndim != 2: return self.reshape_2d(1)[:, sFrom * prodT(self.shape[axisI+1:]) : sTo * prodT(self.shape[axisI+1:])].reshape(retShape) # redirect: use long elements
  chunkSize = int(2e6)
  nChunks = (len(self) + chunkSize - 1) // chunkSize
  if nChunks>1: return concatenate( tuple(self[chunkI*chunkSize : (chunkI+1)*chunkSize, sFrom:sTo] for chunkI in range(nChunks)), 0) # redirect in batches, bc cudamat chokes on big jobs, i.e. jobs with many rows
  if self.shape[0]==1: # then redo as row slice. This also avoids a cudamat limitation (on slicing many columns), sometimes.
   return self.ravel()[sFrom:sTo][newaxis].copy()
  # base case for column slice
  retCm = new_cm(retShape)
  cm_col_slice_read(self.base_shaped(1), sFrom, sTo, retCm)
  return garray(retCm, retShape, None)

 def __iter__(self):
  for i in ranlen(self): yield self[i]
 
 def __setitem__(self, selectors, other):
  # this is different from getitem. There, I can handle the axes one at a time. Here, it's more integrated.
  selectors = nonSeqAsS(selectors)
  for i,sel in enumerate(selectors): # deal with ellipsis
   if sel is Ellipsis: return self.__setitem__(selectors[:i] + t1(slice(None))* (self.ndim - (len(selectors)-1)) + selectors[i+1:], other) # sel==Ellipsis is bad when sel is an array
  if len(selectors) > self.ndim: raise IndexError('more indices than axes')
  if allT( is_array(sel) or isSequence(sel) for sel in selectors) and selectors!=t0:
   if len(selectors)==1:
    if not hasattr(cmType, 'set_selected_columns'):
     raise NotImplementedError("slice assign with a sequence/array as index. Get the newest version of cudamat (or npmat if you're running on the cpu).")
    sel = as_garray(selectors[0])
    if len(sel) != len(other): raise ValueError('number of rows to set != number of provided rows')
    if other.shape[1:] != self.shape[1:]: raise ValueError('shape mismatch in assignment')
    if sel.ndim!=1: raise NotImplementedError('assignment with as index an array of ndim!=1')
    if sel.size==0: return # the current implementation of set_selected_columns doesn't handle that well
    self.base_shaped(1).set_selected_columns(sel.base_shaped(1), other.base_shaped(1))
   else: # >1 selectors, all arrays/sequences. ravel the first dimension of self, and correspondingly unify the first two selectors
    self.reshape(t1(prodT(self.shape[:2])) + self.shape[2:])[t1(as_garray(selectors[0])*self.shape[1]+as_garray(selectors[1])) + selectors[2:]] = as_garray(other)
   return
  if anyT( isSequence(axisSel) or is_array(axisSel) for axisSel in selectors): raise NotImplementedError('slice assign with a sequence/array as index, as well as other indexing objects')
  if anyT( isSlice(axisSel) and axisSel.step not in t2(1, None) for axisSel in selectors): raise NotImplementedError('slice assign with stride != 1')
  if not allT( isNumber(axisSel) or isSlice(axisSel) for axisSel in selectors): raise ValueError('index not understood, in slice assignment.')
  selectors = selectors + t1(slice(None))*(self.ndim-len(selectors))
  # now len(selectors) == ndim, and all selectors are single indices or simple slices
  # task: broadcast other, and do shape check.
  other = as_garray_or_scalar(other)
  assignedShape = tuple( read_simple_slice(axisSel, self.shape[axisI])[2] for axisI, axisSel in enumerate(selectors) if not isNumber(axisSel))
  if is_garray(other):
   if other.ndim < len(assignedShape): other = other.add_axes(len(assignedShape))
   if other.ndim > len(assignedShape):
    if prodT(other.shape[: other.ndim-len(assignedShape)]) != 1: raise ValueError('Incompatible shapes in slice assign: the assigned area has shape %s, and the incoming values have shape %s.' % t2(assignedShape, other.shape))
    other = other.reshape(other.shape[-len(assignedShape):])
   # now other.ndim == len(assignedShape)
   if not allT( other.shape[axisNr] in t2(1, assignedShape[axisNr]) for axisNr in ranlen(assignedShape)):
    raise ValueError('Incompatible shapes in slice assign: the incoming values have shape %s, but the assigned area has shape %s.' % t2(other.shape, assignedShape))
   other = other.tile_to_broadcast(assignedShape)
  # the only time I can use scalar assign is when I don't need cudamat's column assign at all. that only happens when all selectors other than optionally the first are full slices.
  if all2_(selectors[1:], isFullSlice):
   ( cm_row_slice_read(self.base_shaped(1), read_single_index(selectors[0], self.shape[0]), read_single_index(selectors[0], self.shape[0])+1)
     if self.ndim==1 and isNumber(selectors[0]) else
     self[selectors[:1]].base_as_row() # I want this to work even when selectors = t0
     ).assign( other if isNumber(other) else other.base_as_row())
   return
  if isNumber(other): other = garray(other).add_axes(len(assignedShape)).tile_to_broadcast(assignedShape)  
  # now other is a garray of exactly the expected shape, and there are things other than complete slices beyond axis #0 so I'm going to need a col assign.
  # task: get rid of single indices in selectors
  for i in range(self.ndim):
   if isNumber(selectors[i]):
    selectors = modifyT(selectors, i, short_slice(read_single_index(selectors[i], self.shape[i])))
    other = other.reshape(insertT(other.shape, i, t1(1)))
  if not isFullSlice(selectors[0]): return self[selectors[0]].__setitem__(t1(slice(None)) + selectors[1:], other)
  # now all selectors are either full or simple slices; axis 0 is a full slice; and at least one other axis is a simple slice.
  axisI = findIndex3( not isFullSlice(sel) for sel in selectors)
  if all2_(selectors[axisI+1:], isFullSlice): # then do a column slice assign directly using cudamat.
   sFrom, sTo = read_simple_slice(selectors[axisI], self.shape[axisI])[:2]
   elementWidth = prodT(self.shape[axisI+1:])
   if other.size!=0: # cudamat chokes on that
    cm_col_slice_write(self.base_shaped(axisI), sFrom*elementWidth, sTo*elementWidth, other.base_shaped(axisI))
   return
  # remaining case: there are multiple non-full slices, and the slice on axis 0 is full. strategy: transpose to bring one of those non-full slices to the front.
  selfT = self.transpose_simple(axisI)
  selfT[selectors[axisI:] + selectors[:axisI]] = other.transpose_simple(axisI)
  self.base_as_row().assign(selfT.transpose_simple(self.ndim-axisI).base_as_row())

  

 # ------------------------------------------------------------------------------- external, but not for user to see

 def __getstate__(self):
  return t2(self.shape, self.base_as_row().asarray())
 
 def __setstate__(self, state):
  garray.__init__(self, state[1])
  self.set_shape_info(state[0])

 def __array__(self, *dtype):
  envInstruction = _os.environ.get('GNUMPY_IMPLICIT_CONVERSION', 'refuse')
  assert envInstruction in t3('allow', 'warn', 'refuse'), "environment variable GNUMPY_IMPLICIT_CONVERSION, if present, should be one of 'allow', 'warn', 'refuse'."
  if envInstruction=='refuse': raise TypeError("garray objects cannot be quietly converted to numpy arrays, because the environment variable GNUMPY_IMPLICIT_CONVERSION is set to 'refuse', or is not set at all (the default is 'refuse'). Set that variable to 'allow' or 'warn' if you wish to allow quiet conversion. garray's can always be explicitly converted using the .as_numpy_array() method.")
  if envInstruction=='warn': print "gnumpy: warning: a garray object is being quietly converted to a numpy array, and the environment variable GNUMPY_IMPLICIT_CONVERSION is set to 'warn'. garray objects can be explicitly converted using the .as_numpy_array() method."
  return self.as_numpy_array().__array__(*dtype)
  
 def __repr__(self): return self.as_numpy_array().__repr__().replace('array(', 'garray(').replace('\n', '\n ').replace(', dtype=float32', '').replace(', dtype=float64', '') # 64 happens for empty arrays
  
 def __del__(self):
  if not hasattr(self, 'is_alias_of'):
   if isTijmen: print 'gnumpy cleaning up an unfinished garray. mem counting may be off now.'
   return # this object was never finished, because an exception (error or interrupt) occurred in the constructor. This check avoids error messages.
  if self.is_alias_of is None:
   # this is not true in one case: if a reference to self.base is stored somewhere explicitly (somewhere outside self but not in another garray). This happens internally sometimes. I saw it happening on the last line of setitem: a transpose is created (transposes own their mem, are not aliases), and then it's dropped but base (obtained by base_as_row) is still in use for a cm assign call. assert _sys.getrefcount(self.base)==2, _sys.getrefcount(self.base)
   cmsForReuse[self.size].append(self.base)
   if track_memory_usage: memoryUsers[self.allocating_line] = t2(memoryUsers[self.allocating_line][0]-1, memoryUsers[self.allocating_line][1]-self.size*4)
  else:
   assert type(self.is_alias_of).__name__ == 'garray', 'is_alias_of is of unexpected type, of which the str() is: "%s"' % str(type(self.is_alias_of))
   # del self.base # this is only to make the refcount assert not fail



   
castableTypes = numberTypes | set([tuple, list, numpy.array, garray])
