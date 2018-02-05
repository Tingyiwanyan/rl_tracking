import numpy as np
from os.path import join, basename
from glob import glob
from dipy.io.image import load_nifti
from nibabel.streamlines import load
from dipy.viz import actor, window, ui
from dipy.io.gradients import read_bvals_bvecs
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import CsaOdfModel
from dipy.data import get_sphere
from dipy.direction.peaks import peaks_from_model
from dipy.tracking.streamline import transform_streamlines

#dname = '/home/elef/Dropbox/Tingyi/2015_Challenge/'
dname = 'C:\\Users\Tingyi\\Dropbox\\Tingyi\\2015_Challenge\\'

dname_dwi = join(dname, 'ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2')

dname_bundles = join(dname,
                     'ISMRM_2015_Tracto_challenge_ground_truth_bundles_TCK_v2')

dname_rois = join(dname, 'rois')

fbvals = join(dname_dwi, 'NoArtifacts_Relaxation.bvals')
fbvecs = join(dname_dwi, 'NoArtifacts_Relaxation.bvecs')
fdwi = join(dname_dwi, 'NoArtifacts_Relaxation.nii.gz')


def show_sim_data_with_starts_ends():

    labels = np.zeros((180, 216, 180))

    renderer = window.Renderer()

    bundle_names = []

    for fname in glob(join(dname_bundles, '*.tck')):

        base = basename(fname).split('.tck')[0]

        print(base)

        bundle_names.append(base)

        streamlines = load(fname).streamlines

        renderer.add(actor.line(streamlines, colors=np.random.rand(3)))

        head_fname = join(dname_rois, base + '_head.nii.gz')

        data, affine = load_nifti(head_fname)

        labels += data

        #renderer.add(actor.contour_from_roi(data * 200, affine=affine,
        #                                    color=np.random.rand(3),
        #                                    opacity=1))

        tail_fname = join(dname_rois, base + '_tail.nii.gz')

        data, affine = load_nifti(tail_fname)

        labels += data

        #renderer.add(actor.contour_from_roi(data * 200, affine=affine,
        #                                    color=np.random.rand(3),
        #                                    opacity=1))

    # window.show(renderer)

    shape = labels.shape
    image_actor_z = actor.slicer((labels > 0) * 255, affine=affine)

    """
    We can also change also the opacity of the slicer.
    """

    slicer_opacity = 1.0
    image_actor_z.opacity(slicer_opacity)

    """
    We can add additonal slicers by copying the original and adjusting the
    ``display_extent``.
    """

    image_actor_x = image_actor_z.copy()
    image_actor_x.opacity(slicer_opacity)
    x_midpoint = int(np.round(shape[0] / 2))
    image_actor_x.display_extent(x_midpoint,
                                 x_midpoint, 0,
                                 shape[1] - 1,
                                 0,
                                 shape[2] - 1)

    image_actor_y = image_actor_z.copy()
    image_actor_y.opacity(slicer_opacity)
    y_midpoint = int(np.round(shape[1] / 2))
    image_actor_y.display_extent(0,
                                 shape[0] - 1,
                                 y_midpoint,
                                 y_midpoint,
                                 0,
                                 shape[2] - 1)

    renderer.add(image_actor_x)
    renderer.add(image_actor_y)
    renderer.add(image_actor_z)

    show_m = window.ShowManager(renderer, size=(1200, 900),
                                title='DIPY 0.14 Developers\' Edition')
    show_m.initialize()

    """
    After we have initialized the ``ShowManager`` we can go ahead and create
    sliders to move the slices and change their opacity.
    """

    line_slider_z = ui.LineSlider2D(min_value=0,
                                    max_value=shape[2] - 1,
                                    initial_value=shape[2] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_x = ui.LineSlider2D(min_value=0,
                                    max_value=shape[0] - 1,
                                    initial_value=shape[0] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    line_slider_y = ui.LineSlider2D(min_value=0,
                                    max_value=shape[1] - 1,
                                    initial_value=shape[1] / 2,
                                    text_template="{value:.0f}",
                                    length=140)

    opacity_slider = ui.LineSlider2D(min_value=0.0,
                                     max_value=1.0,
                                     initial_value=slicer_opacity,
                                     length=140)

    """
    Now we will write callbacks for the sliders and register them.
    """


    def change_slice_z(i_ren, obj, slider):
        z = int(np.round(slider.value))
        image_actor_z.display_extent(0, shape[0] - 1, 0, shape[1] - 1, z, z)


    def change_slice_x(i_ren, obj, slider):
        x = int(np.round(slider.value))
        image_actor_x.display_extent(x, x, 0, shape[1] - 1, 0, shape[2] - 1)


    def change_slice_y(i_ren, obj, slider):
        y = int(np.round(slider.value))
        image_actor_y.display_extent(0, shape[0] - 1, y, y, 0, shape[2] - 1)


    def change_opacity(i_ren, obj, slider):
        slicer_opacity = slider.value
        image_actor_z.opacity(slicer_opacity)
        image_actor_x.opacity(slicer_opacity)
        image_actor_y.opacity(slicer_opacity)

    line_slider_z.add_callback(line_slider_z.slider_disk,
                               "MouseMoveEvent",
                               change_slice_z)
    line_slider_x.add_callback(line_slider_x.slider_disk,
                               "MouseMoveEvent",
                               change_slice_x)
    line_slider_y.add_callback(line_slider_y.slider_disk,
                               "MouseMoveEvent",
                               change_slice_y)
    opacity_slider.add_callback(opacity_slider.slider_disk,
                                "MouseMoveEvent",
                                change_opacity)

    """
    We'll also create text labels to identify the sliders.
    """


    def build_label(text):
        label = ui.TextBlock2D()
        label.message = text
        label.font_size = 18
        label.font_family = 'Arial'
        label.justification = 'left'
        label.bold = False
        label.italic = False
        label.shadow = False
        label.actor.GetTextProperty().SetBackgroundColor(0, 0, 0)
        label.actor.GetTextProperty().SetBackgroundOpacity(0.0)
        label.color = (1, 1, 1)

        return label


    line_slider_label_z = build_label(text="Z Slice")
    line_slider_label_x = build_label(text="X Slice")
    line_slider_label_y = build_label(text="Y Slice")
    opacity_slider_label = build_label(text="Opacity")

    """
    Now we will create a ``panel`` to contain the sliders and labels.
    """


    panel = ui.Panel2D(center=(1030, 120),
                       size=(300, 200),
                       color=(1, 1, 1),
                       opacity=0.1,
                       align="right")

    panel.add_element(line_slider_label_x, 'relative', (0.1, 0.75))
    panel.add_element(line_slider_x, 'relative', (0.65, 0.8))
    panel.add_element(line_slider_label_y, 'relative', (0.1, 0.55))
    panel.add_element(line_slider_y, 'relative', (0.65, 0.6))
    panel.add_element(line_slider_label_z, 'relative', (0.1, 0.35))
    panel.add_element(line_slider_z, 'relative', (0.65, 0.4))
    panel.add_element(opacity_slider_label, 'relative', (0.1, 0.15))
    panel.add_element(opacity_slider, 'relative', (0.65, 0.2))

    show_m.ren.add(panel)


    global size
    size = renderer.GetSize()


    def win_callback(obj, event):
        global size
        if size != obj.GetSize():
            size_old = size
            size = obj.GetSize()
            size_change = [size[0] - size_old[0], 0]
            panel.re_align(size_change)

    show_m.initialize()


    interactive = True

    renderer.zoom(1.5)
    renderer.reset_clipping_range()

    if interactive:

        show_m.add_window_callback(win_callback)
        show_m.render()
        show_m.start()


def calculate_peaks():

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    data, affine = load_nifti(fdwi)

    maskdata, mask = median_otsu(data, 3, 1, False, vol_idx=range(1, 20))

    tensor_model = TensorModel(gtab, fit_method='WLS')
    tensor_fit = tensor_model.fit(data, mask)

    FA = tensor_fit.fa

    sphere = get_sphere('repulsion724')

    csamodel = CsaOdfModel(gtab, 4) #
    pam = peaks_from_model(model=csamodel,
                           data=maskdata,
                           sphere=sphere,
                           relative_peak_threshold=.5,
                           min_separation_angle=25,
                           mask=mask,
                           return_odf=False,
                           normalize_peaks=True)

    return pam, FA, affine


def get_specific_bundle(bundle_name='UF_right'):

    fname = join(dname_bundles, bundle_name + '.tck')

    base = basename(fname).split('.tck')[0]

    print(base)

    trk_file = load(fname)

    streamlines = trk_file.streamlines
    trk_affine = trk_file.affine

    head_fname = join(dname_rois, base + '_head.nii.gz')

    head, head_affine = load_nifti(head_fname)

    #streamlines2 = transform_streamlines(streamlines,
    #                                     np.linalg.inv(trk_affine))

    print(trk_affine)
    print(head_affine)
    return streamlines, head, head_affine, trk_affine


#show_sim_data_with_starts_ends()

bundle, head, head_affine, trk_affine = get_specific_bundle('UF_right')

pam, FA, affine = calculate_peaks()

bundle = transform_streamlines(bundle,
                               np.linalg.inv(affine))


ren = window.Renderer()

ps = actor.peak_slicer(pam.peak_dirs, pam.peak_values, colors=None)

ren.add(ps)

window.show(ren)

#start = list(bundle[0][0])
#end = list(bundle[0][-1])
starts = bundle[0][0]
ends = bundle[0][-1]

for i in range(len(bundle)-1):
    starts = np.vstack((starts,bundle[i+1][0]))
    ends = np.vstack((ends,bundle[i+1][-1]))

start = np.mean(starts,axis=0)
end = np.mean(ends,axis=0)
start_radius = np.mean(np.std(starts,axis=0))
end_radius = np.mean(np.std(ends,axis=0))


from Rltracking import ReinforcedTracking

rt = ReinforcedTracking(pam.peak_dirs, FA,
                        start, end, start_radius=3,goal_radius=2)

rt.generate_streamlines()

ren.add(actor.line(bundle))
window.show(ren)

bundle2, head, head_affine, trk_affine = get_specific_bundle('UF_left')

bundle2 = transform_streamlines(bundle2,
                                np.linalg.inv(affine))

start2 = list(bundle2[0][0])
end2 = list(bundle2[0][-1])

rt2 = ReinforcedTracking(pam.peak_dirs, FA,
                        start2, end2, start_radius=5, goal_radius=5)

rt2.generate_streamlines()

ren.add(actor.line(bundle2))
window.show(ren)
