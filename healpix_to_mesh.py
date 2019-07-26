import healpy
import stl

def healpix_to_mesh(healpix_map, vmin=None, vmax=None, radius_variation=0.05):
    """Make an STL mesh from a healpix map.

    """
    npix = healpix_map.size
    nside = healpy.npix2nside(npix)

    # May which to cap the maximum variation
    if vmin is None:
        vmin = healpix_map.min()
    if vmax is None:
        vmax = healpix_map.max()

    # For this to be 3D we scale the point radius by this amount
    scale_min = 1 - radius_variation
    scale_max = 1 + radius_variation

    # get the full list of pixel corners
    pix_corners = healpy.boundaries(nside, np.arange(npix))

    # Scale the pixel corners according to the surrounding pixel values,
    # to make this a 3D thing
    for i in range(npix):
        # This is inefficient - vertices are shared between pixels, so
        # we could avoid calculating this scale more than once for each pixel
        # This is probably four times slower than it has to be.
        for j in range(4):
            xyz = all_pix_corners[i,:,j]
            # Use healpy to interpolate the neighbouring pixels to this
            # one to a single scalar value
            theta, phi = healpy.vec2ang(xyz)
            f = healpy.get_interp_val(healpix_map, theta, phi)
            # Convert from intetnsity to radius scaling
            s = scale_min + (scale_max-scale_min)*(f-vmin)/(vmax-vmin)
            # Avoid going beyond the maximum radius variation
            s = np.clip(s, scale_min, scale_max)
            # Scale the vertex
            pix_corners[i,:,j] *= s

    # Make an output mesh object
    mesh = stl.mesh.Mesh(np.zeros(npix*2, dtype=stl.mesh.Mesh.dtype))

    # Divide each pixel into two triangles and assign the vertices
    # of those triangles to the mesh vectors
    for i in range(npix):
        f = 2*i
        mesh.vectors[f,0] = pix_corners[i,:,0]
        mesh.vectors[f,1] = pix_corners[i,:,1]
        mesh.vectors[f,2] = pix_corners[i,:,2]
        f = 2*i + 1
        mesh.vectors[f,0] = pix_corners[i,:,2]
        mesh.vectors[f,1] = pix_corners[i,:,3]
        mesh.vectors[f,2] = pix_corners[i,:,0]

    return mesh


def example():
    T = healpy.read_map('wmap_ilc_9yr_v5.fits')
    T = healpy.ud_grade(T, nside_out=32)
    M = healpix_to_mesh(T)
    M.save('wmap_ilc_9yr_v5.stl')


if __name__ == '__main__':
    example()
