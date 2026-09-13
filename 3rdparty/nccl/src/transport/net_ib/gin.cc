/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#include "gin/gin_host.h"
#include "gin.h"

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

// Check GDR support for GIN. This is run at init, so we don't know yet whether the GPU will support DMA-BUF.
static ncclResult_t ncclGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  if (ncclIbDmaBufSupport(0) == ncclSuccess) return ncclSuccess;

  *gdrSupport = false;
  INFO(NCCL_NET, "Unable to use GIN: Peermem is not supported, nor DMA-BUF.");
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t ncclGinIbGdrGpuSupport(bool gdaki) {
  bool peerMemSupport =
     gdaki ? ncclIbPeerMemSupport() == ncclSuccess : // GDAKI does not support nv_peer_mem.
     ncclIbGdrSupport() == ncclSuccess;
  if (peerMemSupport) return ncclSuccess;

  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  int dmaBufSupportOnDevice = 1;
  CUCHECK(cuDeviceGetAttribute(&dmaBufSupportOnDevice, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, cudaDev));
  if (dmaBufSupportOnDevice == 1) return ncclSuccess;

  WARN("Unable to use GIN: Peermem is not supported, and device %d does not support DMA-BUF.", cudaDev);
  return ncclInvalidUsage;
}

NCCL_PARAM(GinType, "GIN_TYPE", -1);
NCCL_PARAM(GinIbProxyWrBatch, "GIN_IB_PROXY_WR_BATCH", -1);
NCCL_PARAM(GinIbProxySelectiveSignal, "GIN_IB_PROXY_SELECTIVE_SIGNAL", -1);
NCCL_PARAM(GinIbProxyBatchStats, "GIN_IB_PROXY_BATCH_STATS", 0);
NCCL_PARAM(GinIbProxyCqPollBatch, "GIN_IB_PROXY_CQ_POLL_BATCH", 16);

#define NCCL_GIN_IB_PROXY_MAX_WR_BATCH 32
#define NCCL_GIN_IB_PROXY_MAX_CQ_POLL_BATCH 32
#define NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH 8

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;
int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

ncclResult_t ncclGinIbGdakiInit() {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    for (int i = 0; i < ncclNIbDevs; i++) {
      if (ncclIbDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        ncclGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    ncclGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}

extern ncclGin_t ncclGinIb;
extern ncclGin_t ncclGinIbGdaki;
extern ncclGin_t ncclGinIbProxy;

// Initlialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t ncclGinIbInitType(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction, int ginType, ncclGin_t* ginIb) {
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  if (ncclNIbDevs == 0) return ncclInternalError; // Caught in plugin init code, not propagated to user.

  if (ginType == NCCL_GIN_TYPE_GDAKI) goto try_gdaki;
  if (ginType == NCCL_GIN_TYPE_PROXY) goto try_proxy;
  if (ginType != -1) {
    INFO(NCCL_INIT|NCCL_NET, "NET_IB: no support for GIN type %ld", ncclParamGinType());
    return ncclInternalError;
  }

  bool gdrSupport;

  // First try GDAKI
try_gdaki:
  NCCLCHECK(ncclGinIbGdakiInit());
  if (ncclGinIbGdakiNDevs == 0 && ginType == -1) goto try_proxy;
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ true));
  if (!gdrSupport && ginType == -1) goto try_proxy;
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbGdaki, sizeof(ncclGinIb));
  goto end;

  // Then Proxy
try_proxy:
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ false));
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbProxy, sizeof(ncclGinIb));

end:
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}
ncclResult_t ncclGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, ncclParamGinType(), &ncclGinIb);
}

// GIN Entry point, which will then morph into either the GDAKI or PROXY backend
ncclGin_t ncclGinIb = {
  "GIN_IB",
  ncclGinIbInit,
  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};

ncclResult_t ncclGinIbFinalize(void *ctx) {
  if (ctx) free(ctx);
  return ncclIbFinalizeDevices();
}

static ncclResult_t ncclGinIbAllGather(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void *rMhandle = NULL, *sMhandle = NULL;
  void *srequest = NULL, *rrequest = NULL;
  int speer;
  int rpeer;
  void *rbuf;
  int tag;
  int done;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void *)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void *)((uintptr_t)recvBuf + rpeer * len);
      tag = NCCL_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm,
                                      (void *)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, NULL, &srequest),
                      status, out);
      if (rrequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &len,
                                      &tag, &rMhandle, NULL, &rrequest),
                      status, out);
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &done, NULL),
                      status, out);
      if (done)
        rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &done, NULL),
                      status, out);
      if (done)
        srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle)
    ncclNetIb.deregMr(cComm->recvComm, rMhandle);

  if (sMhandle)
    ncclNetIb.deregMr(cComm->sendComm, sMhandle);

  return status;
}

static ncclResult_t ncclGinIbAllToAll(struct ncclGinIbCollComm *cComm, void *src_buf, void *recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;

  void *tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void **)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void *)((uintptr_t)recv_buf + i * len), (void *)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len), len);
  }

out:
  if (tmp_buf)
    free(tmp_buf);

  return status;
}

ncclResult_t ncclGinIbP2PBarrier(struct ncclGinIbCollComm *cComm) {
  // TODO: move allocation to init or use zero-byte allgather
  int *dummy;
  NCCLCHECK(ncclIbMalloc((void **)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

ncclResult_t ncclGinIbConnect(void *ctx, void *handles[], int nranks, int rank,
                              void *listenComm, void **collComm) {
  struct ncclIbListenComm *lComm = (struct ncclIbListenComm *)listenComm;
  struct ncclGinIbCollComm *cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void **)&cCommArray, sizeof(*cCommArray)));

  struct ncclGinIbCollComm *cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do
  {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclNetIb.connect(ctx, lComm->dev, handles[next], &cComm->sendComm, NULL));
    }
    if (cComm->recvComm == NULL)
      NCCLCHECK(ncclNetIb.accept(lComm, &cComm->recvComm, NULL));
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclIbGetGidIndex;
  cComm->dev = lComm->dev;

  cComm->ib.context = ncclIbDevs[cComm->dev].context;
  cComm->ib.pd = ncclIbDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cCommArray = (struct ncclGinIbCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct ncclGinIbCollComm *cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));

  free(cCommArray);
  return ncclSuccess;
}

#include "gdaki/gin_host_gdaki.h"

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI, NULL);
}

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetProperties(int dev, ncclNetProperties_t* props) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (dev >= ncclGinIbGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created", dev, ncclGinIbGdakiNDevs);
    return ncclInvalidUsage;
  }
  NCCLCHECK(ncclIbGetPhysProperties(ncclGinIbGdakiDevIndexes[dev], props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  return ncclNetIb.listen(ctx, ncclGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ true));

  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int dev, void *props))ncclGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, ncclGinConfig_v13_t* config, void **ginCtx, ncclNetDeviceHandle_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;

  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters, config->nContexts, config->queueDepth, config->trafficClass, ginCtx, devHandle));

  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm *)collComm, data, size, type, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct ncclGinIbCollComm *)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiProgress(void *collComm)
{
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t ncclGinIbGdakiQueryLastError(void *ginCtx, bool *hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

ncclGin_t ncclGinIbGdaki = {
  "GIN_IB_GDAKI",
  ncclGinIbGdakiInit,
  ncclGinIbGdakiDevices,
  ncclGinIbGdakiGetProperties,
  ncclGinIbGdakiListen,
  ncclGinIbGdakiConnect,
  ncclGinIbGdakiCreateContext,
  ncclGinIbGdakiRegMrSym,
  NULL, // regMrSymDmaBuf
  ncclGinIbGdakiDeregMrSym,
  ncclGinIbGdakiDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};


struct ncclIbGinProxyMrHandle {
  struct ncclIbMrHandle *mrHandle;
  uintptr_t *base_vas;
  uint32_t *rkeys;
};

ncclResult_t ncclGinIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_PROXY, NULL);
}

ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_t* props) {
  NCCLCHECK(ncclNetIb.getProperties(dev, props));
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyConnect(void *ctx, void *handles[], int nranks, int rank,
                                   void *listenComm, void **collComm) {
  // Check the current GPU supports GDR
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ false));

  // Connect.
  NCCLCHECK(
    ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  return ncclSuccess;
}

struct ncclGinIbProxyBatchStats {
  uint64_t puts;
  uint64_t wrs;
  uint64_t postCalls;
  uint64_t cqes;
  uint64_t cqPollCalls;
  uint64_t cqEmptyPolls;
  uint64_t batches;
  uint64_t selectiveBatches;
  uint64_t maxBatch;
  uint64_t postErrors;
  uint64_t batchHist[NCCL_GIN_IB_PROXY_MAX_WR_BATCH + 1];
};

struct ncclGinIbPendingBatch {
  int count;
  struct ncclIbSendComm* comm;
  struct ncclIbQp* qp;
  struct ncclIbRequest* reqs[NCCL_GIN_IB_PROXY_MAX_WR_BATCH];
  struct ibv_send_wr wr[NCCL_GIN_IB_PROXY_MAX_WR_BATCH];
  struct ibv_sge sge[NCCL_GIN_IB_PROXY_MAX_WR_BATCH];
};

struct ncclGinIbProxyCtx {
  void**        fullRecvComm;
  void**        fullSendComm;
  int rank, nranks;
  int nContexts;
  int queueDepth;
  int wrBatchSize;
  int selectiveSignal;
  int statsEnabled;
  int* sendCqOutstanding;
  int* flushCqOutstanding;
  struct ncclGinIbPendingBatch* pending;
  struct ncclGinIbProxyBatchStats stats;
};

static ncclResult_t ncclGinIbProxyFlushPending(struct ncclGinIbProxyCtx* gc, int rank) {
  struct ncclGinIbPendingBatch* batch = &gc->pending[rank];
  int count = batch->count;
  if (count == 0) return ncclSuccess;

  bool selective = gc->selectiveSignal && count > 1;

  // Selective completion encoding can describe at most eight logical requests
  // per CQE (one 8-bit request slot per wr_id byte). A posted WR chain may be
  // larger than eight: split it into completion groups of <=8 while keeping the
  // entire chain under one ibv_post_send(). This is important for explicit
  // AggregateRequests runs, where the host proxy may drain up to 32 PUT GFDs in
  // one pass without increasing CQE pressure back to one CQE per WR.
  for (int i = 0; i < count; i++) {
    batch->wr[i].next = (i+1 < count) ? &batch->wr[i+1] : NULL;
    batch->wr[i].send_flags = selective ? 0 : IBV_SEND_SIGNALED;
    batch->wr[i].wr_id = selective ? 0 : (uint64_t)(batch->reqs[i] - batch->comm->base.reqs);
  }

  if (selective) {
    for (int groupFirst = 0; groupFirst < count; groupFirst += NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH) {
      int groupCount = ((count - groupFirst) < NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH ? (count - groupFirst) : NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH);
      int ownerIndex = groupFirst + groupCount - 1;
      struct ncclIbRequest* ownerReq = batch->reqs[ownerIndex];
      int ownerSlot = ownerReq - batch->comm->base.reqs;
      uint64_t packedWrId = (uint64_t)(ownerSlot & 0xff);

      // The low byte names the signaled owner; the remaining bytes name the
      // preceding requests in this completion group. Order within the packed id
      // is irrelevant to retirement as every encoded request gets one event
      // decremented when the owner's CQE arrives.
      for (int r = 0; r < groupCount-1; r++) {
        int slot = batch->reqs[groupFirst + r] - batch->comm->base.reqs;
        packedWrId |= (uint64_t)(slot & 0xff) << ((r+1)*8);
      }
      ownerReq->nreqs = groupCount;
      batch->wr[ownerIndex].send_flags = IBV_SEND_SIGNALED;
      batch->wr[ownerIndex].wr_id = packedWrId;
    }
  }

  struct ibv_send_wr* badWr = NULL;
  ncclResult_t res = wrap_ibv_post_send(batch->qp->qp, &batch->wr[0], &badWr);

  if (gc->statsEnabled) {
    gc->stats.wrs += count;
    gc->stats.postCalls++;
    gc->stats.batches++;
    gc->stats.batchHist[count]++;
    if (selective) gc->stats.selectiveBatches++;
    if ((uint64_t)count > gc->stats.maxBatch) gc->stats.maxBatch = count;
    if (res != ncclSuccess) gc->stats.postErrors++;
  }

  if (res != ncclSuccess) {
    if (selective) {
      // Restore completion-group owners so error cleanup/debug paths never leave
      // stale multi-request metadata in reusable request slots.
      for (int groupFirst = 0; groupFirst < count; groupFirst += NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH) {
        int groupCount = ((count - groupFirst) < NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH ? (count - groupFirst) : NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH);
        batch->reqs[groupFirst + groupCount - 1]->nreqs = 1;
      }
    }
    int badIndex = -1;
    if (badWr) {
      for (int i = 0; i < count; i++) {
        if (badWr == &batch->wr[i]) {
          badIndex = i;
          break;
        }
      }
    }
    WARN("GIN/IB/Proxy batch post failed: rank=%d count=%d selective=%d badIndex=%d",
         rank, count, selective ? 1 : 0, badIndex);
    batch->count = 0;
    batch->comm = NULL;
    batch->qp = NULL;
    return res;
  }

  gc->sendCqOutstanding[rank] += count;
  batch->count = 0;
  batch->comm = NULL;
  batch->qp = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclGinIbProxyHandleWc(struct ncclGinIbProxyCtx* gc, int rank,
                                                struct ncclIbNetCommBase* commBase,
                                                struct ncclIbNetCommDevBase* devBase,
                                                const struct ibv_wc* wc,
                                                bool flushCq) {
  struct ncclIbRequest* ownerReq = NULL;
  uint64_t wrId = wc->wr_id;

  if (flushCq) {
    if (wrId >= NET_IB_MAX_REQUESTS) {
      WARN("GIN/IB/Proxy invalid FLUSH completion wr_id=%lu", wrId);
      return ncclInternalError;
    }
    ownerReq = commBase->reqs + wrId;
  } else {
    int ownerSlot = wrId & 0xff;
    if (ownerSlot >= NET_IB_MAX_REQUESTS) {
      WARN("GIN/IB/Proxy invalid completion owner slot=%d wr_id=%lu", ownerSlot, wrId);
      return ncclInternalError;
    }
    ownerReq = commBase->reqs + ownerSlot;
  }

  if (wc->status != IBV_WC_SUCCESS) {
    union ncclSocketAddress addr;
    ncclSocketGetAddr(ownerReq->sock, &addr);
    char localGidString[INET6_ADDRSTRLEN] = "";
    char remoteGidString[INET6_ADDRSTRLEN] = "";
    const char* localGidStr = NULL;
    const char* remoteGidStr = NULL;
    if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      localGidStr = ibvGetGidStr(&devBase->gidInfo.localGid, localGidString, sizeof(localGidString));
      remoteGidStr = ibvGetGidStr(&commBase->remDevs[0].remoteGid, remoteGidString, sizeof(remoteGidString));
    }

    char line[SOCKET_NAME_MAXLEN+1];
    char* hcaName = devBase->pd->context->device->name;
    WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s)%s%s%s%s hca %s",
         ncclSocketToString(&addr, line), wc->status, wc->opcode, wc->byte_len, wc->vendor_err,
         ncclIbReqTypeStr[ownerReq->type], localGidStr ? " localGid ":"", localGidString,
         remoteGidStr ? " remoteGids":"", remoteGidString, hcaName);
    return ncclRemoteError;
  }

  if (flushCq) {
    if (ownerReq->type != NCCL_NET_IB_REQ_FLUSH) {
      WARN("GIN/IB/Proxy FLUSH CQ references non-FLUSH request slot=%lu type=%d", wrId, ownerReq->type);
      return ncclInternalError;
    }
    if (ownerReq->events[0] <= 0) {
      WARN("GIN/IB/Proxy completion for FLUSH request with no pending event: slot=%lu", wrId);
      return ncclInternalError;
    }
    ownerReq->events[0]--;
    if (gc->flushCqOutstanding[rank] <= 0) {
      WARN("GIN/IB/Proxy FLUSH CQ outstanding underflow for rank=%d", rank);
      return ncclInternalError;
    }
    gc->flushCqOutstanding[rank]--;
    return ncclSuccess;
  }

  // Normal operations use one request id. A selective-signaling PUT batch packs
  // up to eight request slots in wr_id, with the low byte naming the owner.
  int nreqs = ownerReq->nreqs;
  if (nreqs < 1 || nreqs > NCCL_GIN_IB_PROXY_MAX_SELECTIVE_BATCH) {
    WARN("GIN/IB/Proxy invalid packed completion nreqs=%d ownerSlot=%d wr_id=%lu",
         nreqs, (int)(wrId & 0xff), wrId);
    return ncclInternalError;
  }

  for (int r = 0; r < nreqs; r++) {
    int slot = (wrId >> (r*8)) & 0xff;
    struct ncclIbRequest* batchReq = commBase->reqs + slot;
    if (nreqs > 1 && batchReq->type != NCCL_NET_IB_REQ_GIN_IPUT) {
      WARN("GIN/IB/Proxy packed completion references non-IPut request slot=%d type=%d wr_id=%lu",
           slot, batchReq->type, wrId);
      return ncclInternalError;
    }
    if (batchReq->events[0] <= 0) {
      WARN("GIN/IB/Proxy request slot=%d has no pending event for wr_id=%lu", slot, wrId);
      return ncclInternalError;
    }
    batchReq->events[0]--;
  }

  if (gc->sendCqOutstanding[rank] < nreqs) {
    WARN("GIN/IB/Proxy send CQ outstanding underflow for rank=%d outstanding=%d completed=%d",
         rank, gc->sendCqOutstanding[rank], nreqs);
    return ncclInternalError;
  }
  gc->sendCqOutstanding[rank] -= nreqs;
  return ncclSuccess;
}

static ncclResult_t ncclGinIbProxyPollCq(struct ncclGinIbProxyCtx* gc, int rank,
                                         struct ncclIbNetCommBase* commBase,
                                         struct ncclIbNetCommDevBase* devBase,
                                         bool flushCq) {
  int pollBatch = (int)ncclParamGinIbProxyCqPollBatch();
  if (pollBatch < 1) pollBatch = 1;
  if (pollBatch > NCCL_GIN_IB_PROXY_MAX_CQ_POLL_BATCH)
    pollBatch = NCCL_GIN_IB_PROXY_MAX_CQ_POLL_BATCH;

  struct ibv_wc wc[NCCL_GIN_IB_PROXY_MAX_CQ_POLL_BATCH];
  int wrDone = 0;
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, pollBatch, wc, &wrDone));

  if (gc->statsEnabled) {
    gc->stats.cqPollCalls++;
    if (wrDone == 0) gc->stats.cqEmptyPolls++;
    gc->stats.cqes += wrDone;
  }

  for (int i = 0; i < wrDone; i++) {
    NCCLCHECK(ncclGinIbProxyHandleWc(gc, rank, commBase, devBase, &wc[i], flushCq));
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinIbProxyProgress(void* ginCtx) {
  struct ncclGinIbProxyCtx* gc = (struct ncclGinIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;

  // Submit PUTs staged by the host proxy first. This preserves the original
  // proxy behavior of making newly drained GFDs visible to the NIC promptly;
  // the CQ pass below can then reap both older completions and, when the NIC is
  // fast enough, completions from WRs posted in this same progress pass.
  for (int c = 0; c < nContexts; c++) {
    if (gc[c].wrBatchSize <= 1) continue;
    for (int rank = 0; rank < nranks; rank++) {
      NCCLCHECK(ncclGinIbProxyFlushPending(&gc[c], rank));
    }
  }

  // Completion discovery is CQ-centric: poll each dedicated GIN send/flush CQ
  // once per progress pass and update request event counters from the returned
  // WC batch. Request objects remain owned/freed by test().
  for (int c = 0; c < nContexts; c++) {
    for (int rank = 0; rank < nranks; rank++) {
      if (gc[c].sendCqOutstanding[rank] > 0 && gc[c].fullSendComm && gc[c].fullSendComm[rank]) {
        struct ncclIbSendComm* sendComm = (struct ncclIbSendComm*)gc[c].fullSendComm[rank];
        NCCLCHECK(ncclGinIbProxyPollCq(&gc[c], rank, &sendComm->base, &sendComm->devs[0].base, false));
      }
      if (gc[c].flushCqOutstanding[rank] > 0 && gc[c].fullRecvComm && gc[c].fullRecvComm[rank]) {
        struct ncclIbRecvComm* recvComm = (struct ncclIbRecvComm*)gc[c].fullRecvComm[rank];
        NCCLCHECK(ncclGinIbProxyPollCq(&gc[c], rank, &recvComm->base, &recvComm->devs[0].base, true));
      }
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclGinIbProxyValidateQueueDepth(struct ncclIbSendComm* comm, int queueDepth) {
  if (queueDepth <= 0) return ncclSuccess;

  struct ibv_qp_attr qpAttr;
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  NCCLCHECK(wrap_ibv_query_qp(comm->base.qps[0].qp, &qpAttr, IBV_QP_CAP, &qpInitAttr));

  if ((int)qpInitAttr.cap.max_send_wr < queueDepth) {
    WARN("GIN_IB_PROXY requested queue depth %d exceeds actual send QP capacity %u",
         queueDepth, qpInitAttr.cap.max_send_wr);
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyCreateContext(void* collComm, ncclGinConfig_v13_t* config, void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  // Make sure all QP we create use the provided traffic class.
  ncclIbSetTrafficClass(cComm->ctx, config->trafficClass);

  // GIN queueDepth is the maximum number of outstanding operations expected per
  // context. The IB proxy request pool is bounded by NET_IB_MAX_REQUESTS, while
  // the send QP is currently created with 2*NET_IB_MAX_REQUESTS WRs. Accept a
  // non-zero queueDepth when the existing proxy resources can satisfy it.
  if (config->queueDepth < 0 || config->queueDepth > NET_IB_MAX_REQUESTS) {
    WARN("GIN_IB_PROXY queue depth %d exceeds supported outstanding request capacity %d",
         config->queueDepth, NET_IB_MAX_REQUESTS);
    return ncclInvalidUsage;
  }

  int nranks;
  struct ncclGinIbProxyCtx* ginProxyCtx = NULL;
  *ginCtx = NULL;
  NCCLCHECK(ncclCalloc(&ginProxyCtx, config->nContexts));
  ginProxyCtx[0].nContexts = config->nContexts;
  ginProxyCtx[0].nranks = nranks = cComm->nranks;

  int wrBatchSize = (int)ncclParamGinIbProxyWrBatch();
  // Keep room for explicit AggregateRequests runs to build a larger WR chain.
  // Ordinary traffic still drains only NCCL_GIN_PROXY_POLL_BATCH descriptors
  // (auto=8) per host pass, so its effective default batch remains eight; an
  // aggregate run may extend to 32 and submit all of them with one post_send.
  if (wrBatchSize < 0) wrBatchSize = NCCL_GIN_IB_PROXY_MAX_WR_BATCH;
  else if (wrBatchSize < 1) wrBatchSize = 1;
  if (wrBatchSize > NCCL_GIN_IB_PROXY_MAX_WR_BATCH) wrBatchSize = NCCL_GIN_IB_PROXY_MAX_WR_BATCH;
  int selectiveSignalParam = (int)ncclParamGinIbProxySelectiveSignal();
  int selectiveSignal = selectiveSignalParam < 0 ? 1 : selectiveSignalParam != 0;
  int statsEnabled = ncclParamGinIbProxyBatchStats() != 0;

  INFO(NCCL_NET, "GIN/IB/Proxy context: requested queueDepth=%d effective request capacity=%d",
       config->queueDepth, NET_IB_MAX_REQUESTS);

  void *lComm = NULL;
  char* handle = NULL, *handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE*cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE*cComm->rank;

  NCCLCHECKGOTO(ncclNetIb.listen(cComm->ctx, cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c=0; c<config->nContexts; c++) {
    struct ncclGinIbProxyCtx* gc = ginProxyCtx+c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void *) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void *) * nranks), ret, end);
    NCCLCHECKGOTO(ncclCalloc(&gc->pending, nranks), ret, end);
    NCCLCHECKGOTO(ncclCalloc(&gc->sendCqOutstanding, nranks), ret, end);
    NCCLCHECKGOTO(ncclCalloc(&gc->flushCqOutstanding, nranks), ret, end);
    gc->rank = cComm->rank;
    gc->nranks = nranks;
    gc->nContexts = config->nContexts;
    gc->queueDepth = config->queueDepth;
    gc->wrBatchSize = wrBatchSize;
    gc->selectiveSignal = selectiveSignal;
    gc->statsEnabled = statsEnabled;

    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.connect(cComm->ctx, cComm->dev, handles+NCCL_NET_HANDLE_MAXSIZE*connectPeer, &gc->fullSendComm[connectPeer], NULL), ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.accept(lComm, &gc->fullRecvComm[acceptPeer], NULL), ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) ||
          (gc->fullRecvComm[acceptPeer] == NULL));
      NCCLCHECKGOTO(ncclGinIbProxyValidateQueueDepth(
                        (struct ncclIbSendComm*)gc->fullSendComm[connectPeer], config->queueDepth), ret, end);
      NCCLCHECKGOTO(ncclGinIbP2PBarrier(cComm), ret, end);
    }
  }

end:
  free(handles);
  if (lComm) ncclNetIb.closeListen(lComm);
  if (ret != ncclSuccess) free(ginProxyCtx);
  else *ginCtx = ginProxyCtx;
  return ret;
}

ncclResult_t ncclGinIbProxyDestroyContext(void* ginCtx) {
  struct ncclGinIbProxyCtx* gc = (struct ncclGinIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;

  if (gc[0].statsEnabled) {
    uint64_t puts = 0, wrs = 0, postCalls = 0, cqes = 0;
    uint64_t cqPollCalls = 0, cqEmptyPolls = 0;
    uint64_t batches = 0, selectiveBatches = 0, maxBatch = 0, postErrors = 0;
    uint64_t batchHist[NCCL_GIN_IB_PROXY_MAX_WR_BATCH + 1] = {0};
    for (int c = 0; c < nContexts; c++) {
      puts += gc[c].stats.puts;
      wrs += gc[c].stats.wrs;
      postCalls += gc[c].stats.postCalls;
      cqes += gc[c].stats.cqes;
      cqPollCalls += gc[c].stats.cqPollCalls;
      cqEmptyPolls += gc[c].stats.cqEmptyPolls;
      batches += gc[c].stats.batches;
      selectiveBatches += gc[c].stats.selectiveBatches;
      postErrors += gc[c].stats.postErrors;
      for (int i = 1; i <= NCCL_GIN_IB_PROXY_MAX_WR_BATCH; i++)
        batchHist[i] += gc[c].stats.batchHist[i];
      if (gc[c].stats.maxBatch > maxBatch) maxBatch = gc[c].stats.maxBatch;
    }
    double cqesPerPoll = cqPollCalls ? (double)cqes / (double)cqPollCalls : 0.0;
    double emptyPollRate = cqPollCalls ? 100.0 * (double)cqEmptyPolls / (double)cqPollCalls : 0.0;
    INFO(NCCL_NET,
         "GIN/IB/Proxy batch stats: puts=%lu wrs=%lu postCalls=%lu cqes=%lu batches=%lu selectiveBatches=%lu maxBatch=%lu postErrors=%lu",
         puts, wrs, postCalls, cqes, batches, selectiveBatches, maxBatch, postErrors);
    INFO(NCCL_NET,
         "GIN/IB/Proxy CQ stats: pollCalls=%lu emptyPolls=%lu emptyPollRate=%.2f%% cqesPerPoll=%.3f",
         cqPollCalls, cqEmptyPolls, emptyPollRate, cqesPerPoll);

    uint64_t singleBatches = batchHist[1];
    uint64_t multiBatches = batches >= singleBatches ? batches - singleBatches : 0;
    uint64_t batchedPuts = puts >= singleBatches ? puts - singleBatches : 0;
    int fullBatchSize = gc[0].wrBatchSize;
    uint64_t fullBatches =
      (fullBatchSize >= 1 && fullBatchSize <= NCCL_GIN_IB_PROXY_MAX_WR_BATCH) ? batchHist[fullBatchSize] : 0;
    double avgBatch = postCalls ? (double)wrs / (double)postCalls : 0.0;
    double multiBatchRate = batches ? 100.0 * (double)multiBatches / (double)batches : 0.0;
    double batchedPutRate = puts ? 100.0 * (double)batchedPuts / (double)puts : 0.0;
    double fullBatchRate = batches ? 100.0 * (double)fullBatches / (double)batches : 0.0;

    INFO(NCCL_NET,
         "GIN/IB/Proxy batch summary: avgBatch=%.3f multiBatchRate=%.2f%% batchedPutRate=%.2f%% fullBatchSize=%d fullBatchRate=%.2f%%",
         avgBatch, multiBatchRate, batchedPutRate, fullBatchSize, fullBatchRate);
    for (int i = 1; i <= NCCL_GIN_IB_PROXY_MAX_WR_BATCH; i++) {
      if (batchHist[i] != 0)
        INFO(NCCL_NET, "GIN/IB/Proxy batch histogram: size=%d batches=%lu puts=%lu",
             i, batchHist[i], batchHist[i] * (uint64_t)i);
    }
  }

  for (int c=0; c<nContexts; c++) {
    if (gc[c].pending) {
      for (int i = 0; i < nranks; i++) {
        if (gc[c].pending[i].count != 0)
          WARN("GIN/IB/Proxy destroying context with %d unflushed PUTs for rank %d",
               gc[c].pending[i].count, i);
      }
    }

    if (gc[c].fullRecvComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i=0; i<nranks; i++) {
        NCCLCHECK(ncclNetIb.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }

    free(gc[c].pending);
    gc[c].pending = NULL;
    free(gc[c].sendCqOutstanding);
    gc[c].sendCqOutstanding = NULL;
    free(gc[c].flushCqOutstanding);
    gc[c].flushCqOutstanding = NULL;
  }
  free(gc);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle;
  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  NCCLCHECKNOWARN(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags, (void **)&ginMrHandle->mrHandle), NCCL_NET);

  NCCLCHECK(ncclCalloc(&ginMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&ginMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, ginMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &ginMrHandle->mrHandle->mrs[0]->rkey, ginMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;

  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type, uint64_t mr_flags, void** mhandle, void **ginHandle) {
  return ncclGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;

  NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle));
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyCloseColl(void* collComm) {
  free(collComm);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPut(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle, size_t size,
                                uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;

  void *srcPtr = (void *)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
  void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->nreqs = 1;
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  if (ginProxyCtx->statsEnabled) ginProxyCtx->stats.puts++;

  // Batch size 1 is the original v2.30.4 fast path.
  if (ginProxyCtx->wrBatchSize <= 1) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    struct ibv_sge sge;
    memset(&sge, 0, sizeof(sge));

    wr.opcode                  = IBV_WR_RDMA_WRITE;
    wr.send_flags              = IBV_SEND_SIGNALED;
    wr.wr_id                   = req - comm->base.reqs;
    wr.next                    = NULL;
    wr.wr.rdma.remote_addr     = (uint64_t)dstPtr;
    wr.wr.rdma.rkey            = rkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    sge.addr = (uintptr_t)srcPtr;
    sge.length = size;
    sge.lkey = lkey;

    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
    ncclIbAddEvent(req, qp->devIndex);
    ginProxyCtx->sendCqOutstanding[rank]++;

    if (ginProxyCtx->statsEnabled) {
      ginProxyCtx->stats.wrs++;
      ginProxyCtx->stats.postCalls++;
      ginProxyCtx->stats.batches++;
      ginProxyCtx->stats.batchHist[1]++;
      if (ginProxyCtx->stats.maxBatch < 1) ginProxyCtx->stats.maxBatch = 1;
    }

    *request = req;
    return ncclSuccess;
  }

  struct ncclGinIbPendingBatch* batch = &ginProxyCtx->pending[rank];
  if (batch->count == 0) {
    batch->comm = comm;
    batch->qp = qp;
  } else if (batch->comm != comm || batch->qp != qp) {
    NCCLCHECK(ncclGinIbProxyFlushPending(ginProxyCtx, rank));
    batch->comm = comm;
    batch->qp = qp;
  }

  int idx = batch->count;
  if (idx >= ginProxyCtx->wrBatchSize || idx >= NCCL_GIN_IB_PROXY_MAX_WR_BATCH) {
    NCCLCHECK(ncclGinIbProxyFlushPending(ginProxyCtx, rank));
    idx = 0;
    batch->comm = comm;
    batch->qp = qp;
  }

  batch->reqs[idx] = req;
  memset(&batch->wr[idx], 0, sizeof(batch->wr[idx]));
  memset(&batch->sge[idx], 0, sizeof(batch->sge[idx]));

  batch->wr[idx].opcode = IBV_WR_RDMA_WRITE;
  batch->wr[idx].wr.rdma.remote_addr = (uint64_t)dstPtr;
  batch->wr[idx].wr.rdma.rkey = rkey;
  batch->wr[idx].sg_list = &batch->sge[idx];
  batch->wr[idx].num_sge = 1;

  batch->sge[idx].addr = (uintptr_t)srcPtr;
  batch->sge[idx].length = size;
  batch->sge[idx].lkey = lkey;

  // Mark the logical request outstanding before returning it to the host proxy.
  // The pending WR list is submitted by ginProgress() at the end of this pass.
  ncclIbAddEvent(req, qp->devIndex);
  batch->count++;

  *request = req;

  if (batch->count >= ginProxyCtx->wrBatchSize)
    NCCLCHECK(ncclGinIbProxyFlushPending(ginProxyCtx, rank));

  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIGet(void *ginCtx, int context, uint64_t remoteOffset, void *remoteMhandle,
                                 size_t size, uint64_t localOffset, void *localMhandle, uint32_t rank,
                                 void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];
  if (ginProxyCtx->wrBatchSize > 1) NCCLCHECK(ncclGinIbProxyFlushPending(ginProxyCtx, rank));

  struct ncclIbGinProxyMrHandle *remoteMrHandle = (struct ncclIbGinProxyMrHandle *)remoteMhandle;
  struct ncclIbGinProxyMrHandle *localMrHandle = (struct ncclIbGinProxyMrHandle *)localMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->nreqs = 1;
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->iget.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  void *remotePtr = (void *)(remoteMrHandle->base_vas[rank] + remoteOffset);
  void *localPtr = (void *)(localMrHandle->base_vas[ginProxyCtx->rank] + localOffset);
  uint32_t rkey = remoteMrHandle->rkeys[rank];
  uint32_t lkey = localMrHandle->mrHandle->mrs[0]->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode                  = IBV_WR_RDMA_READ;
  wr.send_flags              = IBV_SEND_SIGNALED; // TODO: Potentially optimize this?
  wr.wr_id                   = req - comm->base.reqs;
  wr.next                    = NULL;
  wr.wr.rdma.remote_addr     = (uint64_t)remotePtr;
  wr.wr.rdma.rkey            = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)localPtr;
  sge.length = size;
  sge.lkey = lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);
  ginProxyCtx->sendCqOutstanding[rank]++;

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPutSignal(void *ginCtx, int context, uint64_t srcOff, void *srcMhandle,
                                      size_t size, uint64_t dstOff, void *dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void *signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, void **request) {
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("ncclGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];
  if (ginProxyCtx->wrBatchSize > 1) NCCLCHECK(ncclGinIbProxyFlushPending(ginProxyCtx, rank));

  struct ncclIbGinProxyMrHandle *srcMrHandle = (struct ncclIbGinProxyMrHandle *)srcMhandle;
  struct ncclIbGinProxyMrHandle *dstMrHandle = (struct ncclIbGinProxyMrHandle *)dstMhandle;
  struct ncclIbGinProxyMrHandle *signalMrHandle = (struct ncclIbGinProxyMrHandle *)signalMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ncclIbQp *qp = &comm->base.qps[0];
  int devIndex = qp->devIndex;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->nreqs = 1;
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL
  if (size > 0 && dstMrHandle) {
    void *srcPtr = (void *)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
    void *dstPtr = (void *)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->mrHandle->mrs[0]->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    // PUT
    wr[0].opcode                  = IBV_WR_RDMA_WRITE;
    wr[0].send_flags              = 0; // We only need the CQE from the signal
    wr[0].wr_id                   = req - comm->base.reqs;
    wr[0].next                    = &wr[1];
    wr[0].wr.rdma.remote_addr     = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey            = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;  // Local buffer address
    sge[0].length = size;  // Size of the transfer
    sge[0].lkey = lkey;  // Local key
  }

  void *signalPtr = (void *)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  // SIGNAL
  wr[1].opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags              = IBV_SEND_SIGNALED;
  wr[1].wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  wr[1].next                    = NULL;
  wr[1].wr.atomic.remote_addr   = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey          = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;

  // Send the put and the signal in one go
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, size > 0 ? &wr[0] : &wr[1], &bad_wr));
  ncclIbAddEvent(req, qp->devIndex);
  ginProxyCtx->sendCqOutstanding[rank]++;
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyTest(void* collComm, void *request, int *done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  *done = 0;

  // CQ polling is performed by ncclGinIbProxyProgress(). test() only owns
  // request retirement, which keeps request lifetime separate from completion
  // discovery and avoids repeatedly polling the same CQ per outstanding GFD.
  for (int d = 0; d < NCCL_IB_MAX_DEVS_PER_NIC; d++) {
    if (req->events[d] != 0) return ncclSuccess;
  }

  *done = 1;
  NCCLCHECK(ncclIbFreeRequest(req));
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIFlush(void *ginCtx, int context, void* mhandle, uint32_t rank, void **request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];
  if (ginProxyCtx->wrBatchSize > 1) NCCLCHECK(ncclGinIbProxyFlushPending(ginProxyCtx, rank));
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  struct ncclIbGinProxyMrHandle *ginMrHandle = (struct ncclIbGinProxyMrHandle *)mhandle;
  struct ncclIbQp *qp = &comm->devs[0].gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->nreqs = 1;
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->iput.rank = rank;
  req->ginProxyCtx = ginProxyCtx;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  void *flushPtr = (void *)(ginMrHandle->base_vas[rank]);
  wr.wr.rdma.remote_addr = (uint64_t)flushPtr;
  wr.wr.rdma.rkey = ginMrHandle->rkeys[rank];
  wr.sg_list = &comm->devs[qp->devIndex].gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  TRACE(NCCL_NET, "NET/IB: %s: Posting a flush request (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);
  TIME_START(4);
  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp->qp, &wr, &bad_wr));
  TIME_STOP(4);

  ncclIbAddEvent(req, qp->devIndex);
  ginProxyCtx->flushCqOutstanding[rank]++;

  TRACE(NCCL_NET, "NET/IB: %s: Flush request posted (req=%p, comm=%p, wr_id=%ld)", __func__, req, req->base, wr.wr_id);

  *request = req;
  return ncclSuccess;
}

// No support for NCCL_IB_SPLIT_DATA_ON_QPS or NCCL_IB_MERGE_NICS
ncclGin_t ncclGinIbProxy = {
  "GIN_IB_PROXY",
  ncclGinIbProxyInit,
  ncclIbDevices,
  ncclGinIbProxyGetProperties,
  ncclIbListen,
  ncclGinIbProxyConnect,
  ncclGinIbProxyCreateContext,
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  ncclGinIbProxyDestroyContext,
  ncclGinIbCloseColl,
  ncclIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  ncclGinIbProxyIGet,
  ncclGinIbProxyIFlush,
  ncclGinIbProxyTest,
  ncclGinIbProxyProgress,
  NULL,
  ncclGinIbFinalize
};
