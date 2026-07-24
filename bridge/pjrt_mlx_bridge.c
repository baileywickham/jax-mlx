// jax-mlx PJRT bridge: implements the PJRT C API and forwards compile /
// execute / buffer traffic to Python (runtime.dispatch) via the CPython API.
// The plugin runs inside the host CPython process; -undefined dynamic_lookup
// resolves Py* symbols at load time.
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pjrt_c_api.h"

// ---------- errors ----------
typedef struct { int code; char msg[1024]; } Error;

static PJRT_Error* err_new(int code, const char* msg) {
  Error* e = calloc(1, sizeof(Error));
  e->code = code;
  snprintf(e->msg, sizeof(e->msg), "%s", msg ? msg : "(null)");
  return (PJRT_Error*)e;
}
static void Bridge_Error_Destroy(PJRT_Error_Destroy_Args* a) { free(a->error); }
static void Bridge_Error_Message(PJRT_Error_Message_Args* a) {
  Error* e = (Error*)a->error;
  a->message = e->msg; a->message_size = strlen(e->msg);
}
static PJRT_Error* Bridge_Error_GetCode(PJRT_Error_GetCode_Args* a) {
  a->code = (PJRT_Error_Code)((Error*)a->error)->code; return NULL;
}
// Our errors never carry payloads: call the visitor zero times and return
// success. Required because callers (jaxlib) invoke this to report/log any
// PJRT_Error -- including errors returned by unimplemented functions such as
// PJRT_Client_TopologyDescription. Leaving this slot on the generic
// UNIMPLEMENTED filler causes infinite recursion: reporting the "unimplemented"
// error from ForEachPayload itself requires calling ForEachPayload on the new
// error, ad infinitum, which stack-overflows (observed as SIGSEGV).
static PJRT_Error* Bridge_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* a) {
  (void)a; return NULL;
}

// ---------- always-ready events ----------
static PJRT_Event* event_new(void) { return (PJRT_Event*)calloc(1, 8); }
static PJRT_Error* Bridge_Event_Destroy(PJRT_Event_Destroy_Args* a) { free(a->event); return NULL; }
static PJRT_Error* Bridge_Event_IsReady(PJRT_Event_IsReady_Args* a) { a->is_ready = true; return NULL; }
static PJRT_Error* Bridge_Event_Error(PJRT_Event_Error_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_Event_Await(PJRT_Event_Await_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_Event_OnReady(PJRT_Event_OnReady_Args* a) {
  a->callback(NULL, a->user_arg); return NULL;
}

// ---------- python dispatch ----------
static PyObject* g_dispatch = NULL;
static int64_t g_shlo_cur[3] = {1, 0, 0};
static const int64_t g_shlo_min[3] = {0, 9, 0};

__attribute__((visibility("default")))
void JaxMlxInstallDispatcher(PyObject* fn) { Py_XINCREF(fn); g_dispatch = fn; }

__attribute__((visibility("default")))
void JaxMlxSetStablehloVersion(long a, long b, long c) {
  g_shlo_cur[0] = a; g_shlo_cur[1] = b; g_shlo_cur[2] = c;
}

// Steals `args` (a new-ref tuple or NULL). On success stores a new ref in *out.
static PJRT_Error* call_py(const char* method, PyObject* args, PyObject** out) {
  if (!g_dispatch) return err_new(13, "jax-mlx: dispatcher not installed");
  PyGILState_STATE st = PyGILState_Ensure();
  // call_py can run during exception unwinding (e.g. a PJRT buffer being
  // destroyed while an execute-time NotImplementedError is still propagating
  // up through the interpreter). Calling into Python with an exception
  // already set on the thread state makes CPython raise a spurious
  // SystemError ("... returned a result with an exception set") the moment
  // the dispatched code touches anything that itself makes a C-level call
  // (e.g. acquiring a threading.Lock in a `with` block), which then looks
  // like *our* call failed. Stash any pending exception before dispatching
  // and put it back afterwards so call_py is safe to invoke unconditionally.
  PyObject *pending_t, *pending_v, *pending_tb;
  PyErr_Fetch(&pending_t, &pending_v, &pending_tb);
  PJRT_Error* err = NULL;
  PyObject* res = PyObject_CallFunction(
      g_dispatch, "sO", method, args ? args : Py_None);
  if (!res) {
    PyObject *t, *v, *tb;
    PyErr_Fetch(&t, &v, &tb);
    PyObject* s = v ? PyObject_Str(v) : NULL;
    err = err_new(13, s ? PyUnicode_AsUTF8(s) : "python error");
    Py_XDECREF(s); Py_XDECREF(t); Py_XDECREF(v); Py_XDECREF(tb);
  } else {
    *out = res;
  }
  if (pending_t) PyErr_Restore(pending_t, pending_v, pending_tb);
  Py_XDECREF(args);
  PyGILState_Release(st);
  return err;
}

// ---------- singletons: client / device / description / memory ----------
static int g_client_o, g_device_o, g_desc_o, g_memory_o;
#define CLIENT ((PJRT_Client*)&g_client_o)
#define DEVICE ((PJRT_Device*)&g_device_o)
#define DESC   ((PJRT_DeviceDescription*)&g_desc_o)
#define MEMORY ((PJRT_Memory*)&g_memory_o)
static PJRT_Device* g_devices[1];
static PJRT_Memory* g_memories[1];

static PJRT_Error* Bridge_Plugin_Initialize(PJRT_Plugin_Initialize_Args* a) { (void)a; return NULL; }

static PJRT_NamedValue g_attrs[2];
static PJRT_Error* Bridge_Plugin_Attributes(PJRT_Plugin_Attributes_Args* a) {
  g_attrs[0] = (PJRT_NamedValue){ .struct_size = PJRT_NamedValue_STRUCT_SIZE,
      .name = "stablehlo_current_version", .name_size = 25,
      .type = PJRT_NamedValue_kInt64List,
      .int64_array_value = g_shlo_cur, .value_size = 3 };
  g_attrs[1] = (PJRT_NamedValue){ .struct_size = PJRT_NamedValue_STRUCT_SIZE,
      .name = "stablehlo_minimum_version", .name_size = 25,
      .type = PJRT_NamedValue_kInt64List,
      .int64_array_value = g_shlo_min, .value_size = 3 };
  a->attributes = g_attrs; a->num_attributes = 2;
  return NULL;
}

static PJRT_Error* Bridge_Client_Create(PJRT_Client_Create_Args* a) {
  g_devices[0] = DEVICE; g_memories[0] = MEMORY;
  a->client = CLIENT; return NULL;
}
static PJRT_Error* Bridge_Client_Destroy(PJRT_Client_Destroy_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_Client_PlatformName(PJRT_Client_PlatformName_Args* a) {
  a->platform_name = "mlx"; a->platform_name_size = 3; return NULL;
}
static PJRT_Error* Bridge_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* a) {
  a->process_index = 0; return NULL;
}
static PJRT_Error* Bridge_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* a) {
  a->platform_version = "jax-mlx 0.1.0"; a->platform_version_size = 13; return NULL;
}
static PJRT_Error* Bridge_Client_Devices(PJRT_Client_Devices_Args* a) {
  a->devices = g_devices; a->num_devices = 1; return NULL;
}
static PJRT_Error* Bridge_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* a) {
  a->addressable_devices = g_devices; a->num_addressable_devices = 1; return NULL;
}
static PJRT_Error* Bridge_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* a) {
  a->addressable_memories = g_memories; a->num_addressable_memories = 1; return NULL;
}
static PJRT_Error* Bridge_Client_LookupDevice(PJRT_Client_LookupDevice_Args* a) {
  if (a->id != 0) return err_new(3, "jax-mlx: only device 0 exists");
  a->device = DEVICE; return NULL;
}
static PJRT_Error* Bridge_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* a) {
  if (a->local_hardware_id != 0) return err_new(3, "jax-mlx: only device 0");
  a->addressable_device = DEVICE; return NULL;
}
static PJRT_Error* Bridge_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* a) {
  for (size_t i = 0; i < a->default_assignment_size; ++i) a->default_assignment[i] = 0;
  return NULL;
}

static PJRT_Error* Bridge_Device_GetDescription(PJRT_Device_GetDescription_Args* a) {
  a->device_description = DESC; return NULL;
}
static PJRT_Error* Bridge_Device_IsAddressable(PJRT_Device_IsAddressable_Args* a) {
  a->is_addressable = true; return NULL;
}
static PJRT_Error* Bridge_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* a) {
  a->local_hardware_id = 0; return NULL;
}
static PJRT_Error* Bridge_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* a) {
  a->memories = g_memories; a->num_memories = 1; return NULL;
}
static PJRT_Error* Bridge_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* a) {
  a->memory = MEMORY; return NULL;
}
// Not in the brief's original source: jaxlib's PjRtCApiDevice constructor
// calls this unconditionally via InitAttributes() and treats any returned
// PJRT_Error as fatal (LogFatalIfPjrtError aborts the process), so this
// cannot be left on the generic UNIMPLEMENTED filler. We report zero
// attributes and a no-op deleter for the (unused, NULL) opaque handle.
static void Bridge_Device_Attributes_Deleter(PJRT_Device_Attributes* d) { (void)d; }
static PJRT_Error* Bridge_Device_GetAttributes(PJRT_Device_GetAttributes_Args* a) {
  a->attributes = NULL; a->num_attributes = 0;
  a->device_attributes = NULL;
  a->attributes_deleter = Bridge_Device_Attributes_Deleter;
  return NULL;
}

static PJRT_Error* Bridge_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* a) {
  a->id = 0; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* a) { a->process_index = 0; return NULL; }
static PJRT_Error* Bridge_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* a) {
  a->attributes = NULL; a->num_attributes = 0; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* a) {
  a->device_kind = "mlx"; a->device_kind_size = 3; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* a) {
  a->debug_string = "MlxDevice(id=0)"; a->debug_string_size = 15; return NULL;
}
static PJRT_Error* Bridge_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* a) {
  a->to_string = "MlxDevice(id=0)"; a->to_string_size = 15; return NULL;
}

static PJRT_Error* Bridge_Memory_Id(PJRT_Memory_Id_Args* a) { a->id = 0; return NULL; }
static PJRT_Error* Bridge_Memory_Kind(PJRT_Memory_Kind_Args* a) {
  a->kind = "device"; a->kind_size = 6; return NULL;
}
static PJRT_Error* Bridge_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* a) {
  a->kind_id = 1; return NULL;
}
static PJRT_Error* Bridge_Memory_DebugString(PJRT_Memory_DebugString_Args* a) {
  a->debug_string = "mlx unified memory"; a->debug_string_size = 18; return NULL;
}
static PJRT_Error* Bridge_Memory_ToString(PJRT_Memory_ToString_Args* a) {
  a->to_string = "device"; a->to_string_size = 6; return NULL;
}
static PJRT_Error* Bridge_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* a) {
  a->devices = g_devices; a->num_devices = 1; return NULL;
}

// ---------- buffers ----------
static const size_t kDtypeBytes[] = {0, 1, 1, 2, 4, 8, 1, 2, 4, 8, 2, 4, 8, 2, 8, 16};

typedef struct {
  int64_t id; int dtype; size_t ndim; int64_t dims[8]; size_t nbytes;
  int64_t minor_to_major[8];
} Buf;

static Buf* buf_new(int64_t id, int dtype, size_t ndim, const int64_t* dims) {
  Buf* b = calloc(1, sizeof(Buf));
  b->id = id; b->dtype = dtype; b->ndim = ndim;
  size_t n = 1;
  for (size_t i = 0; i < ndim; ++i) { b->dims[i] = dims[i]; n *= (size_t)dims[i]; }
  b->nbytes = n * kDtypeBytes[dtype];
  for (size_t i = 0; i < ndim; ++i) b->minor_to_major[i] = (int64_t)(ndim - 1 - i);
  return b;
}

static PJRT_Error* Bridge_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* a) {
  if (a->num_dims > 8) return err_new(3, "jax-mlx: rank > 8 unsupported");
  if (a->type >= (int)(sizeof(kDtypeBytes)/sizeof(*kDtypeBytes)) ||
      kDtypeBytes[a->type] == 0)
    return err_new(12, "jax-mlx: unsupported buffer dtype");
  if (a->num_byte_strides != 0) {
    // Only dense row-major accepted in v0; jax sends dense for np arrays.
    size_t expect = kDtypeBytes[a->type];
    for (size_t i = a->num_dims; i-- > 0;) {
      if ((size_t)a->byte_strides[i] != expect)
        return err_new(12, "jax-mlx: non-dense host strides unsupported");
      expect *= (size_t)a->dims[i];
    }
  }
  size_t n = kDtypeBytes[a->type];
  for (size_t i = 0; i < a->num_dims; ++i) n *= (size_t)a->dims[i];

  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* bytes = PyBytes_FromStringAndSize((const char*)a->data, (Py_ssize_t)n);
  PyObject* dims = PyTuple_New((Py_ssize_t)a->num_dims);
  for (size_t i = 0; i < a->num_dims; ++i)
    PyTuple_SET_ITEM(dims, (Py_ssize_t)i, PyLong_FromLongLong(a->dims[i]));
  PyObject* type_obj = PyLong_FromLong(a->type);
  PyObject* args = PyTuple_Pack(3, bytes, type_obj, dims);
  Py_DECREF(bytes); Py_DECREF(type_obj); Py_DECREF(dims);
  PyGILState_Release(st);

  PyObject* res = NULL;
  PJRT_Error* err = call_py("buffer_from_host", args, &res);
  if (err) return err;
  st = PyGILState_Ensure();
  int64_t id = PyLong_AsLongLong(res);
  Py_DECREF(res);
  PyGILState_Release(st);

  a->buffer = (PJRT_Buffer*)buf_new(id, a->type, a->num_dims, a->dims);
  a->done_with_host_buffer = event_new();
  return NULL;
}

static PJRT_Error* Bridge_Buffer_Destroy(PJRT_Buffer_Destroy_Args* a) {
  Buf* b = (Buf*)a->buffer;
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* id_obj = PyLong_FromLongLong(b->id);
  PyObject* args = PyTuple_Pack(1, id_obj);
  Py_DECREF(id_obj);
  PyGILState_Release(st);
  PyObject* res = NULL;
  PJRT_Error* err = call_py("buffer_delete", args, &res);
  st = PyGILState_Ensure();
  Py_XDECREF(res);
  PyGILState_Release(st);
  free(b);
  // PJRT destructors are documented as infallible -- jaxlib's caller
  // (LogFatalIfPjrtError) treats any non-NULL PJRT_Error from Destroy as a
  // fatal CHECK failure and aborts the process. If the Python-side delete
  // failed (most likely because we're already unwinding a pending exception
  // from a prior compile/execute error), swallow the error instead of
  // propagating it: the Python registry entry is popped defensively
  // (`_buffers.pop(id, None)`), so a missed delete here leaks at most one
  // registry entry rather than crashing the whole process.
  if (err) free(err);  // Error* is a plain calloc'd struct; safe to free directly.
  return NULL;
}

static PJRT_Error* Bridge_Buffer_ElementType(PJRT_Buffer_ElementType_Args* a) {
  a->type = (PJRT_Buffer_Type)((Buf*)a->buffer)->dtype; return NULL;
}
static PJRT_Error* Bridge_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* a) {
  Buf* b = (Buf*)a->buffer; a->dims = b->dims; a->num_dims = b->ndim; return NULL;
}
static PJRT_Error* Bridge_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* a) {
  Buf* b = (Buf*)a->buffer;
  a->unpadded_dims = b->dims; a->num_dims = b->ndim; return NULL;
}
static PJRT_Error* Bridge_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* a) {
  a->on_device_size_in_bytes = ((Buf*)a->buffer)->nbytes; return NULL;
}
static PJRT_Error* Bridge_Buffer_Device(PJRT_Buffer_Device_Args* a) {
  a->device = DEVICE; return NULL;
}
static PJRT_Error* Bridge_Buffer_Memory(PJRT_Buffer_Memory_Args* a) {
  a->memory = MEMORY; return NULL;
}
static PJRT_Error* Bridge_Buffer_Delete(PJRT_Buffer_Delete_Args* a) {
  (void)a; return NULL;  // actual free happens in Destroy
}
static PJRT_Error* Bridge_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* a) {
  a->is_deleted = false; return NULL;
}
static PJRT_Error* Bridge_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* a) {
  a->is_on_cpu = false; return NULL;
}
static PJRT_Error* Bridge_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* a) {
  a->event = event_new(); return NULL;
}

static PJRT_Error* Bridge_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* a) {
  Buf* b = (Buf*)a->buffer;
  a->layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  a->layout.extension_start = NULL;
  a->layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
  a->layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  a->layout.tiled.extension_start = NULL;
  a->layout.tiled.minor_to_major = b->minor_to_major;
  a->layout.tiled.minor_to_major_size = b->ndim;
  a->layout.tiled.tile_dims = NULL;
  a->layout.tiled.tile_dim_sizes = NULL;
  a->layout.tiled.num_tiles = 0;
  return NULL;
}

static PJRT_Error* Bridge_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* a) {
  Buf* b = (Buf*)a->src;
  if (a->dst == NULL) { a->dst_size = b->nbytes; return NULL; }
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* id_obj = PyLong_FromLongLong(b->id);
  PyObject* args = PyTuple_Pack(1, id_obj);
  Py_DECREF(id_obj);
  PyGILState_Release(st);
  PyObject* res = NULL;
  PJRT_Error* err = call_py("buffer_to_host", args, &res);
  if (err) return err;
  st = PyGILState_Ensure();
  char* data = NULL; Py_ssize_t len = 0;
  if (PyBytes_AsStringAndSize(res, &data, &len) != 0) {
    Py_DECREF(res); PyGILState_Release(st);
    return err_new(13, "jax-mlx: buffer_to_host returned invalid bytes");
  }
  if ((size_t)len > a->dst_size) {
    Py_DECREF(res); PyGILState_Release(st);
    return err_new(13, "jax-mlx: host buffer too small");
  }
  memcpy(a->dst, data, (size_t)len);
  Py_DECREF(res);
  PyGILState_Release(st);
  a->event = event_new();
  return NULL;
}

// ---------- executables ----------
typedef struct {
  int64_t id;
  size_t num_outputs;
  int out_dtypes[64];
  size_t out_ndims[64];
  int64_t out_dims[64][8];
  // Flattened view of out_dims/out_ndims, built once at Compile time so
  // Bridge_Executable_OutputDimensions can hand back pointers into this
  // per-Exec storage (genuinely live as long as the executable itself).
  int64_t flat_dims[64 * 8];
  size_t dim_sizes[64];
} Exec;

static PJRT_Error* Bridge_Client_Compile(PJRT_Client_Compile_Args* a) {
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* code = PyBytes_FromStringAndSize(
      a->program->code, (Py_ssize_t)a->program->code_size);
  PyObject* args = PyTuple_Pack(1, code);
  Py_DECREF(code);
  PyGILState_Release(st);

  PyObject* res = NULL;
  PJRT_Error* err = call_py("compile", args, &res);
  if (err) return err;

  st = PyGILState_Ensure();
  Exec* e = calloc(1, sizeof(Exec));
  PyObject* id_obj = PyTuple_GetItem(res, 0);      // borrowed
  e->id = PyLong_AsLongLong(id_obj);
  PyObject* specs = PyTuple_GetItem(res, 1);        // borrowed
  Py_ssize_t nspecs = PyList_Size(specs);
  e->num_outputs = (size_t)nspecs;
  PJRT_Error* bounds_err = NULL;
  if (nspecs > 64) {
    bounds_err = err_new(3, "jax-mlx: executable has more than 64 outputs");
    nspecs = 64;
    e->num_outputs = 64;
  }
  size_t k = 0;
  for (Py_ssize_t i = 0; i < nspecs; ++i) {
    PyObject* spec = PyList_GetItem(specs, i);          // borrowed
    PyObject* dtype_obj = PyTuple_GetItem(spec, 0);      // borrowed
    e->out_dtypes[i] = (int)PyLong_AsLong(dtype_obj);
    PyObject* dims = PyTuple_GetItem(spec, 1);           // borrowed
    Py_ssize_t ndim = PyTuple_Size(dims);
    if (ndim > 8) {
      // Rank > 8 is not supported: report a clean error instead of silently
      // truncating (matches the num_outputs>64 handling below).
      if (!bounds_err)
        bounds_err = err_new(3, "jax-mlx: output rank > 8 unsupported");
      ndim = 8;
    }
    e->out_ndims[i] = (size_t)ndim;
    e->dim_sizes[i] = (size_t)ndim;
    for (Py_ssize_t j = 0; j < ndim; ++j) {
      PyObject* d = PyTuple_GetItem(dims, j);            // borrowed
      int64_t v = PyLong_AsLongLong(d);
      e->out_dims[i][j] = v;
      e->flat_dims[k++] = v;
    }
  }
  Py_DECREF(res);
  PyGILState_Release(st);
  if (bounds_err) { free(e); return bounds_err; }
  a->executable = (PJRT_LoadedExecutable*)e;
  return NULL;
}

static PJRT_Error* Bridge_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* a) {
  Exec* e = (Exec*)a->executable;
  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* id_obj = PyLong_FromLongLong(e->id);
  PyObject* args = PyTuple_Pack(1, id_obj);
  Py_DECREF(id_obj);
  PyGILState_Release(st);
  PyObject* res = NULL;
  PJRT_Error* err = call_py("executable_delete", args, &res);
  st = PyGILState_Ensure();
  Py_XDECREF(res);
  PyGILState_Release(st);
  free(e);
  // Same infallible-destructor pattern as Bridge_Buffer_Destroy: jaxlib's
  // LogFatalIfPjrtError treats any non-NULL PJRT_Error from a destructor as
  // fatal, so swallow errors from the Python-side delete instead of
  // propagating them -- a missed executable_delete leaks at most one
  // `_executables` registry entry (`_executables.pop(id, None)` is
  // defensive), rather than crashing the whole process.
  if (err) free(err);
  return NULL;
}
static PJRT_Error* Bridge_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* a) {
  a->executable = (PJRT_Executable*)a->loaded_executable; return NULL;
}
static PJRT_Error* Bridge_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* a) {
  a->addressable_devices = g_devices; a->num_addressable_devices = 1; return NULL;
}
// Not in the brief's original source: jaxlib's PjRtCApiLoadedExecutable
// constructor calls this unconditionally via InitAddressableDeviceLogicalIds()
// and treats any returned PJRT_Error as fatal (LogFatalIfPjrtError aborts the
// process, observed as SIGABRT), so this cannot be left on the generic
// UNIMPLEMENTED filler. Single replica/partition -> one (0, 0) entry.
static PJRT_LogicalDeviceIds g_logical_ids[1] = {{0, 0}};
static PJRT_Error* Bridge_LoadedExecutable_AddressableDeviceLogicalIds(
    PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args* a) {
  a->addressable_device_logical_ids = g_logical_ids;
  a->num_addressable_device_logical_ids = 1;
  return NULL;
}
// Not in the brief's original source: jaxlib's PjRtCApiLoadedExecutable
// constructor also calls this unconditionally via InitDeviceAssignment(),
// fatal-on-error like the two above. It expects a serialized
// xla.DeviceAssignmentProto (replica_count=1, computation_count=1,
// computation_devices=[{replica_device_ids: [0]}]), hand-encoded here since
// we have no protobuf runtime linked into the bridge:
//   field 1 (varint)  tag 0x08 value 1   -- replica_count
//   field 2 (varint)  tag 0x10 value 1   -- computation_count
//   field 3 (message) tag 0x1A len 3     -- computation_devices[0]
//     nested field 1 (packed varint) tag 0x0A len 1 value 0  -- replica_device_ids
static const unsigned char g_device_assignment[] = {
    0x08, 0x01, 0x10, 0x01, 0x1A, 0x03, 0x0A, 0x01, 0x00};
static void Bridge_DeviceAssignment_Deleter(PJRT_DeviceAssignmentSerialized* d) {
  (void)d;
}
static PJRT_Error* Bridge_LoadedExecutable_GetDeviceAssignment(
    PJRT_LoadedExecutable_GetDeviceAssignment_Args* a) {
  a->serialized_bytes = (const char*)g_device_assignment;
  a->serialized_bytes_size = sizeof(g_device_assignment);
  a->serialized_device_assignment = NULL;
  a->serialized_device_assignment_deleter = Bridge_DeviceAssignment_Deleter;
  return NULL;
}
static PJRT_Error* Bridge_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* a) { (void)a; return NULL; }
static PJRT_Error* Bridge_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* a) { a->is_deleted = false; return NULL; }

// PJRT_Executable and PJRT_LoadedExecutable alias the same Exec* here
// (GetExecutable just casts the pointer through, it doesn't allocate a
// second object). So Executable_Destroy must be a no-op: freeing happens
// exactly once, in LoadedExecutable_Destroy. If Executable_Destroy also
// freed, jaxlib's normal teardown sequence (destroy the PJRT_Executable
// view, then destroy the PJRT_LoadedExecutable) would double-free.
static PJRT_Error* Bridge_Executable_Destroy(PJRT_Executable_Destroy_Args* a) {
  (void)a; return NULL;  // freed via LoadedExecutable_Destroy
}
static PJRT_Error* Bridge_Executable_Name(PJRT_Executable_Name_Args* a) {
  a->executable_name = "jax_mlx_exec"; a->executable_name_size = 12; return NULL;
}
static PJRT_Error* Bridge_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* a) {
  a->num_replicas = 1; return NULL;
}
static PJRT_Error* Bridge_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* a) { a->num_partitions = 1; return NULL; }
static PJRT_Error* Bridge_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* a) {
  a->num_outputs = ((Exec*)a->executable)->num_outputs; return NULL;
}
static PJRT_Error* Bridge_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* a) {
  Exec* e = (Exec*)a->executable;
  a->output_types = (PJRT_Buffer_Type*)e->out_dtypes;
  a->num_output_types = e->num_outputs;
  return NULL;
}
// dims/dim_sizes are flattened once into Exec at Compile time (see
// Bridge_Client_Compile), so the pointers handed back here point into
// per-Exec storage that genuinely lives as long as the executable itself
// (as opposed to file-scope statics shared -- and clobbered -- across Execs).
static PJRT_Error* Bridge_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* a) {
  Exec* e = (Exec*)a->executable;
  a->num_outputs = e->num_outputs;
  a->dims = e->flat_dims;
  a->dim_sizes = e->dim_sizes;
  return NULL;
}
static PJRT_Error* Bridge_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args* a) {
  a->executable_fingerprint = ""; a->executable_fingerprint_size = 0; return NULL;
}

static PJRT_Error* Bridge_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* a) {
  Exec* e = (Exec*)a->executable;
  if (a->num_devices != 1)
    return err_new(12, "jax-mlx: multi-device execution unsupported");

  PyGILState_STATE st = PyGILState_Ensure();
  PyObject* ids = PyTuple_New((Py_ssize_t)a->num_args);
  for (size_t i = 0; i < a->num_args; ++i)
    PyTuple_SET_ITEM(ids, (Py_ssize_t)i,
        PyLong_FromLongLong(((Buf*)a->argument_lists[0][i])->id));
  PyObject* eid = PyLong_FromLongLong(e->id);
  PyObject* args = PyTuple_Pack(2, eid, ids);
  Py_DECREF(eid);
  Py_DECREF(ids);
  PyGILState_Release(st);

  PyObject* res = NULL;
  PJRT_Error* err = call_py("execute", args, &res);
  if (err) return err;

  st = PyGILState_Ensure();
  Py_ssize_t n = PyList_Size(res);
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* t = PyList_GetItem(res, i);               // borrowed
    PyObject* id_obj = PyTuple_GetItem(t, 0);             // borrowed
    int64_t id = PyLong_AsLongLong(id_obj);
    PyObject* dtype_obj = PyTuple_GetItem(t, 1);          // borrowed
    int dtype = (int)PyLong_AsLong(dtype_obj);
    PyObject* dims = PyTuple_GetItem(t, 2);               // borrowed
    size_t ndim = (size_t)PyTuple_Size(dims);
    if (ndim > 8) {
      // Rank > 8 is not supported: cdims is int64_t[8] and Buf.dims is [8],
      // so passing ndim through unclamped would overflow both. Return a
      // clean error instead of truncating.
      Py_DECREF(res);
      PyGILState_Release(st);
      return err_new(3, "jax-mlx: output rank > 8 unsupported");
    }
    int64_t cdims[8];
    for (size_t j = 0; j < ndim && j < 8; ++j) {
      PyObject* d = PyTuple_GetItem(dims, (Py_ssize_t)j); // borrowed
      cdims[j] = PyLong_AsLongLong(d);
    }
    a->output_lists[0][i] = (PJRT_Buffer*)buf_new(id, dtype, ndim, cdims);
  }
  Py_DECREF(res);
  PyGILState_Release(st);

  if (a->device_complete_events) a->device_complete_events[0] = event_new();
  return NULL;
}

// ---------- api table ----------
static PJRT_Error* GenericUnimplemented(void* a) {
  (void)a; return err_new(12, "jax-mlx: PJRT function not implemented");
}

static PJRT_Api g_api;

__attribute__((visibility("default")))
const PJRT_Api* GetPjrtApi(void) {
  memset(&g_api, 0, sizeof(g_api));
  g_api.struct_size = PJRT_Api_STRUCT_SIZE;
  g_api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
  g_api.pjrt_api_version.major_version = PJRT_API_MAJOR;
  g_api.pjrt_api_version.minor_version = PJRT_API_MINOR;
  // Fill every function slot with a safe UNIMPLEMENTED handler, then override.
  for (void** p = (void**)&g_api.PJRT_Error_Destroy;
       p < (void**)((char*)&g_api + sizeof(g_api)); ++p)
    *p = (void*)GenericUnimplemented;

#define SET(name) g_api.PJRT_##name = Bridge_##name
  SET(Error_Destroy); SET(Error_Message); SET(Error_GetCode);
  SET(Error_ForEachPayload);
  SET(Event_Destroy); SET(Event_IsReady); SET(Event_Error);
  SET(Event_Await); SET(Event_OnReady);
  SET(Plugin_Initialize); SET(Plugin_Attributes);
  SET(Client_Create); SET(Client_Destroy); SET(Client_PlatformName);
  SET(Client_ProcessIndex); SET(Client_PlatformVersion); SET(Client_Devices);
  SET(Client_AddressableDevices); SET(Client_AddressableMemories);
  SET(Client_LookupDevice); SET(Client_LookupAddressableDevice);
  SET(Client_DefaultDeviceAssignment);
  SET(Device_GetDescription); SET(Device_IsAddressable);
  SET(Device_LocalHardwareId); SET(Device_AddressableMemories);
  SET(Device_DefaultMemory); SET(Device_GetAttributes);
  SET(DeviceDescription_Id); SET(DeviceDescription_ProcessIndex);
  SET(DeviceDescription_Attributes); SET(DeviceDescription_Kind);
  SET(DeviceDescription_DebugString); SET(DeviceDescription_ToString);
  SET(Memory_Id); SET(Memory_Kind); SET(Memory_Kind_Id);
  SET(Memory_DebugString); SET(Memory_ToString); SET(Memory_AddressableByDevices);
  SET(Client_BufferFromHostBuffer);
  SET(Buffer_Destroy); SET(Buffer_ElementType); SET(Buffer_Dimensions);
  SET(Buffer_UnpaddedDimensions); SET(Buffer_OnDeviceSizeInBytes);
  SET(Buffer_Device); SET(Buffer_Memory); SET(Buffer_Delete);
  SET(Buffer_IsDeleted); SET(Buffer_IsOnCpu); SET(Buffer_ReadyEvent);
  SET(Buffer_ToHostBuffer); SET(Buffer_GetMemoryLayout);
  SET(Client_Compile);
  SET(LoadedExecutable_Destroy); SET(LoadedExecutable_GetExecutable);
  SET(LoadedExecutable_AddressableDevices); SET(LoadedExecutable_Delete);
  SET(LoadedExecutable_IsDeleted); SET(LoadedExecutable_Execute);
  SET(LoadedExecutable_AddressableDeviceLogicalIds);
  SET(LoadedExecutable_GetDeviceAssignment);
  SET(Executable_Destroy); SET(Executable_Name); SET(Executable_NumReplicas);
  SET(Executable_NumPartitions); SET(Executable_NumOutputs);
  SET(Executable_OutputElementTypes); SET(Executable_OutputDimensions);
  SET(Executable_Fingerprint);
#undef SET
  return &g_api;
}
