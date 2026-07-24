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
  return err;
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
  char* data; Py_ssize_t len;
  PyBytes_AsStringAndSize(res, &data, &len);
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
#undef SET
  return &g_api;
}
