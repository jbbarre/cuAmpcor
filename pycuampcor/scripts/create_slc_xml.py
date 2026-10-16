import os

import click
import isce
import isceobj


@click.command()
@click.argument('id1', type=str)
@click.argument('id1_range', type=int)
@click.argument('id2', type=str)
@click.argument('id2_range', type=int)
@click.option('--byte-order', type=click.Choice(['b', 'l']), default='b', show_default=True,
              help='Byte order of the SLC files (b=big-endian, GAMMA-native FCOMPLEX; l=little-endian).')
def create_slc_xml(id1:str, id1_range:int, id2:str, id2_range:int, byte_order:str):
    """Create XML and VRT files for SLC images.

    Args:
        id1 (str): Filename of the first SLC image without extension.
        id1_range (int): Width of the first SLC image.
        id2 (str): Filename of the second SLC image without extension.
        id2_range (int): Width of the second SLC image.
        byte_order (str): Byte order of both SLC files ('b' or 'l'), recorded in the XML/VRT
            so that cuDenseOffsets.py picks it up by default.
    """

    for fname, width in [(id1 + '.slc', id1_range), (id2 + '.slc', id2_range)]:
        filesize = os.path.getsize(fname)
        bytes_per_pixel = 8   # complex64 SLC: 4 bytes real + 4 bytes imag
        length = filesize // (width * bytes_per_pixel)

        img = isceobj.createSlcImage()
        img.setFilename(fname)
        img.setWidth(width)
        img.setLength(length)
        img.setByteOrder(byte_order)
        img.setAccessMode('READ')
        img.renderHdr()   # writes fname.xml
        img.renderVRT()   # writes fname.vrt

if __name__ == '__main__':
    create_slc_xml()    


