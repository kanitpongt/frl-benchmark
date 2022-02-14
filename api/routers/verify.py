import time
from enum import Enum
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from deepface import DeepFace as df

class VerifyLibrary(str, Enum):
    deepface: "DeepFace"
    insight_face: "InsightFace"


class DeepFaceRecognitionModel(str, Enum):
    vgg_face: "VGG-Face"
    facenet: "Facenet"
    facenet512: "Facenet512"
    openface: "OpenFace"
    deepface: "DeepFace"
    deepId: "DeepID"
    arcface: "ArcFace"
    dlib: "Dlib"


class DeepFaceDetectorModel(str, Enum):
    opencv: "opencv"
    dlib: "dlib"
    ssd: "ssd"
    retinaface: "retinaface"
    mtcnn: "mtcnn"


class DeepFaceMetric(str, Enum):
    cos: "cosine"
    euclid: "euclidean"
    euclid_norm: "euclidean_l2"


class VerifyResponse(BaseModel):
    distance: float
    max_verification_threshold: float
    seconds: float
    result: bool


class VerifyRequest(BaseModel):
    images: List[str] = []
    library: VerifyLibrary


class DeepFaceVerifyResponse(VerifyResponse):
    detector_model_used: DeepFaceRecognitionModel

class DeepFaceVerifyRequest(VerifyRequest):
    detector_backend: DeepFaceDetectorModel
    model_name: DeepFaceRecognitionModel
    distance_metric: DeepFaceMetric

DEFAULT_LIBRARY: VerifyLibrary = "deepface"

router = APIRouter()

@router.post('/', response_model=VerifyResponse)
def verify_face(req: VerifyRequest) -> VerifyResponse:
    if req.library == VerifyLibrary.deepface:
        return deepface_verify(req)

@router.post('/deepface', response_model=DeepFaceVerifyResponse)
def deepface_verify(req: DeepFaceVerifyRequest) -> DeepFaceVerifyResponse:
    # Verify request
    if "model_name" in list(req.keys()):
        model_name = req["model_name"]

    if "distance_metric" in list(req.keys()):
        distance_metric = req["distance_metric"]

    if "detector_backend" in list(req.keys()):
        detector_backend = req["detector_backend"]

    instances = req["img"]

    if len(instances) < 2:
        return {'success': False, 'error': 'you must pass at least two img in your request'}
    
    tic = time.time()
    resp_obj = deepface_verify_wrapper(images=instances, model= model_name, detector=detector_backend, metric=distance_metric)
    toc = time.time()
    
    resp_obj["detector_model_used"] = model_name
    resp_obj["seconds"] = toc-tic
    
    return resp_obj

def insight_verify(images, model, distance, detector):
    return "TODO"

def deepface_verify_wrapper(images: List[str], model: DeepFaceDetectorModel="VGG-Face", detector: DeepFaceDetectorModel="opencv", metric: DeepFaceMetric="cosine"):
    try:
        return df.verify(images['img1'], images['img2'], model_name=model, distance_metric=metric, detector_backend=detector)
    except Exception as err:
        return {'success': False, 'error': str(err)}