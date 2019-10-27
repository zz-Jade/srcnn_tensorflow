import numpy as np

#全部转为一维数组进行评价,转成n*1维数组

def psnr(y_true,y_pred):
    diff_square = 0
    for i in range(np.size(y_true)):
        diff_square = diff_square + (y_true[i]-y_pred[i]) * (y_true[i]-y_pred[i])
    return 10*np.log10(255*255*np.size(y_true)*np.size(y_pred)/diff_square)

def ssim(y_true,y_pred):
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    pred_variance = np.var(y_pred)
    true_variance = np.var(y_true)
    covariance = covariance1(y_true,y_pred)
    print((2*pred_mean*true_mean+0.01),(2*covariance+0.02),(pred_mean*pred_mean+true_mean*true_mean+0.01),pred_variance*pred_variance+true_variance*true_variance+0.02)
    ssim1 = (2*pred_mean*true_mean+0.01)*(2*covariance+0.02)/((pred_mean*pred_mean+true_mean*true_mean+0.01)*(pred_variance*pred_variance+true_variance*true_variance+0.02))
    return ssim1

def covariance1(y_true,y_pred):
    a = y_true - y_true.mean()
    b = y_pred - y_pred.mean()
    covariance = 0
    for i in range(np.size(y_true)):
        covariance = covariance+a[i]*b[i]
    covariance = covariance/(np.size(a)+np.size(b)-1)
    return covariance


