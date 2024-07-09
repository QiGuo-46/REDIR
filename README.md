# REDIR
The employment of the event-based synthetic aperture imaging (E-SAI)
technique, which has the capability to capture high-frequency light intensity varia-
tions, has facilitated its extensive application on scene de-occlusion reconstruction
tasks. However, existing methods usually require prior information and have strict
restriction of camera motion on SAI acquisition methods. This paper proposes a
novel end-to-end refocus-free variable E-SAI de-occlusion image reconstruction
approach REDIR, which can align the global and local features of the variable
event data and effectively achieve high-resolution imaging of pure event streams.
To further improve the reconstruction of the occluded target, we propose a percep-
tual mask-gated connection module to interlink information between modules, and
incorporate a spatial-temporal attention mechanism into the SNN block to enhance
target extraction ability of the model. Through extensive experiments, our model
achieves state-of-the-art reconstruction quality on the traditional E-SAI dataset,
while verifying the effectiveness of the variable event data feature registration
method on our newly introduced V-ESAI dataset, which obviates the reliance
on prior knowledge and extends the applicability of SAI acquisition methods by
incorporating focus changes, lens rotations, and non-uniform motion.

<img src="VERE-Net-Archicture.pdf" height="200">

## V-ESAI Dataset 
https://drive.google.com/file/d/1Sx_pikaAg--ix6W1_AdNzsVa4rRZovXF/view?usp=drive_link
