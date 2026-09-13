"""차량 인스턴스 세그멘테이션 학습·평가 패키지.

프로젝트: Segmentation 기반 도로 CCTV 의 도로 대비 차량 면적 비를 활용한 교통 혼잡도 측정.

이 패키지는 모델 계열(YOLO / Mask R-CNN / Mask2Former / RF-DETR)이 서로 달라도
 - 같은 데이터(car_seg_split)를
 - 같은 인터페이스(Predictor)로 예측하고
 - 같은 채점기(COCOeval, 마스크 기준 mAP)로 평가해
실험 기록표에 나란히 비교할 수 있게 한다.
"""

__version__ = "0.1.0"

CLASS_NAMES = ("car", "bus", "truck")  # 학습용 0-based 클래스 번호 (YOLO 규약과 동일)
