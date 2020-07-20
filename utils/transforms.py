import numpy as np
import cv2
from albumentations.core.transforms_interface import DualTransform

import copy

class RandomMaskSafeShiftX(DualTransform):
    def __init__(self, shift_limit=1., always_apply=False, p=0.5):
        super(DualTransform, self).__init__(always_apply=always_apply, p=p)
        if isinstance(shift_limit, float):
            assert 0. <= abs(shift_limit) <= 1.
            self.shift_limit = np.random.uniform(low=-abs(shift_limit), high=abs(shift_limit))
        elif isinstance(shift_limit, tuple) or isinstance(shift_limit, list):
            assert all(list(map(lambda x: isinstance(x, float), shift_limit)))
            assert all(list(map(lambda x: 0. <= abs(x) <= 1., shift_limit)))
            assert shift_limit[0] < shift_limit[1]
            self.shift_limit = np.random.uniform(low=shift_limit[0], high=shift_limit[1])
        else:
            raise ValueError

    def apply(self, img, **params):
        safe_shift = params['safe_shift']
        shift = int(img.shape[1] * self.shift_limit)
        if shift < safe_shift[0]:
            shift = safe_shift[0]
        elif shift > safe_shift[1]:
            shift = safe_shift[1]
        img = np.roll(img, shift=shift, axis=1)
        return img

    def apply_to_mask(self, mask, **params):
        safe_shift = params['safe_shift']
        shift = int(mask.shape[1] * self.shift_limit)
        if shift < safe_shift[0]:
            shift = safe_shift[0]
        elif shift > safe_shift[1]:
            shift = safe_shift[1]
        mask = np.roll(mask, shift=shift, axis=1)
        return mask

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']
        x_len = mask.shape[1]
        safe_shift_r = x_len - mask.nonzero()[1].max() - 65
        safe_shift_l = - mask.nonzero()[1].min()
        return {'safe_shift': (safe_shift_l, safe_shift_r)}

    @property
    def targets_as_params(self):
        return ["image", "mask"]


class RandomMaskBlackOut(DualTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(DualTransform, self).__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        mask = params['mask_']
        del params['mask_']
        defects_idxs = params['defects_to_black_out_idxs']

        for defect_idx in defects_idxs:
            ch, comp_idx = defect_idx
            img[cv2.connectedComponents(mask[:, :, ch].astype(np.uint8))[1] == comp_idx] = 0
        return img

    def apply_to_mask(self, mask, **params):
        defects_idxs = params['defects_to_black_out_idxs']

        for defect_idx in defects_idxs:
            ch, comp_idx = defect_idx
            mask[cv2.connectedComponents(mask[:, :, ch].astype(np.uint8))[1] == comp_idx, ch] = 0
            cv2.imshow('aa', mask[:, :, ch:ch+1])
            cv2.waitKey(0)
        return mask

    def get_params_dependent_on_targets(self, params):
        mask = params['mask']
        defects_idxs = []
        for ch in range(4):
            n_components, _ = cv2.connectedComponents(mask[:, :, ch].astype(np.uint8))
            for component_idx in range(1, n_components):
                defects_idxs.append((ch, component_idx))

        temp = np.empty(len(defects_idxs), dtype=object)
        temp[:] = defects_idxs
        defects_idxs = temp
        del temp

        n_defects_to_black_out = 1 if len(defects_idxs) == 1 else np.random.randint(low=1, high=len(defects_idxs))
        n_defects_to_black_out = 3
        defects_to_black_out_idxs = np.random.choice(defects_idxs, size=n_defects_to_black_out, replace=False)
        print(defects_to_black_out_idxs)
        return {'defects_to_black_out_idxs': defects_to_black_out_idxs, 'mask_': mask}

    @property
    def targets_as_params(self):
        return ["image", "mask"]