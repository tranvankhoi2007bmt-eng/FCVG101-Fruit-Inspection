"""Classical image processing and camera geometry (OpenCV BGR convention)."""
import cv2
import numpy as np


def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    return img


def pinhole(points, K):
    points = np.asarray(points, dtype=float)
    if np.any(points[:, 2] <= 0):
        raise ValueError('Points must be in front of the camera.')
    q = (K @ points.T).T
    return q[:, :2] / q[:, 2:3]


def filters(image):
    return {'Box': cv2.blur(image, (5, 5)),
            'Gaussian': cv2.GaussianBlur(image, (5, 5), 0),
            'Median': cv2.medianBlur(image, 5)}


def edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {'Sobel': cv2.normalize(cv2.magnitude(dx, dy), None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8),
            'Canny': cv2.Canny(gray, 50, 150)}


def frequency(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    f = np.fft.fftshift(np.fft.fft2(gray))
    y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
    radius = max(1, min(gray.shape) // 12)
    mask = (y-gray.shape[0]//2)**2 + (x-gray.shape[1]//2)**2 <= radius**2
    low = np.fft.ifft2(np.fft.ifftshift(f * mask)).real
    high = np.fft.ifft2(np.fft.ifftshift(f * ~mask)).real
    return {'Spectrum': np.log1p(abs(f)), 'Low pass': np.clip(low, 0, 255).astype(np.uint8),
            'High pass magnitude': cv2.normalize(abs(high), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)}


def features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    hmax = cv2.dilate(h, None)
    ys, xs = np.where((h == hmax) & (h > max(0, 0.01*h.max())))
    marked = image.copy()
    for x, y in zip(xs, ys):
        cv2.circle(marked, (int(x), int(y)), 2, (0, 0, 255), -1)
    out, counts = {'Harris': marked}, {'Harris': int(len(xs))}
    for name, detector in [('SIFT', cv2.SIFT_create(nfeatures=1000)),
                           ('ORB', cv2.ORB_create(nfeatures=1000))]:
        kp = detector.detect(gray, None)
        out[name] = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
        counts[name] = len(kp)
    return out, counts


def match_views(first, second, method='ORB', ratio=0.75):
    detector = cv2.ORB_create(nfeatures=1500) if method == 'ORB' else cv2.SIFT_create(nfeatures=1500)
    norm = cv2.NORM_HAMMING if method == 'ORB' else cv2.NORM_L2
    k1, d1 = detector.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = detector.detectAndCompute(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), None)
    good = []
    if d1 is not None and d2 is not None and len(d2) >= 2:
        pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio*p[1].distance]
    inliers = None
    if len(good) >= 4:
        a = np.float32([k1[m.queryIdx].pt for m in good])
        b = np.float32([k2[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.sum())
    visual = cv2.drawMatches(first, k1, second, k2, good[:50], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return visual, {'keypoints_first': len(k1), 'keypoints_second': len(k2),
                    'ratio_matches': len(good), 'ransac_inliers': inliers}


def calibrate(images, pattern=(9, 6), square_size=25.0):
    if not images or square_size <= 0 or min(pattern) < 2:
        raise ValueError('Supply valid checkerboard images and dimensions.')
    obj = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)*square_size
    objects, corners, accepted = [], [], []
    size = images[0].shape[1::-1]
    for i, img in enumerate(images):
        if img.shape[1::-1] != size:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, pts = cv2.findChessboardCornersSB(gray, pattern)
        if ok:
            objects.append(obj.copy()); corners.append(pts); accepted.append(i)
    if len(objects) < 5:
        raise ValueError(f'Only {len(objects)} usable views; at least 5 required.')
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, corners, size, None, None)
    errors = []
    for objp, observed, r, t in zip(objects, corners, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, r, t, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((observed-projected)**2, axis=2)))))
    preview = images[accepted[0]].copy()
    cv2.drawChessboardCorners(preview, pattern, corners[0], True)
    return {'K': K, 'distortion': dist.ravel(), 'rms_px': float(rms), 'errors_px': errors,
            'accepted': accepted, 'size': size}, preview


def undistort(image, K, dist, size=None):
    if size and tuple(image.shape[1::-1]) != tuple(size):
        raise ValueError('Use the same camera resolution as calibration.')
    return cv2.undistort(image, K, dist)
