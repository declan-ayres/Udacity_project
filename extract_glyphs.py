import os.path
import fontforge
import psMat
import PIL
from PIL import Image
import numpy as np

filehandle = open('allfonts-paths-from-server', 'r')
files_list = filehandle.readlines()
path = 'glyphs_output/'
excluding_list = ['/usr/share/ghostscript/9.10/Resource/CIDFSubst/DroidSansFallback.ttf', '/usr/share/texlive/texmf-dist/fonts/truetype/public/dejavu/DejaVuSans-BoldOblique.ttf','/usr/share/texlive/texmf-dist/fonts/truetype/public/gentium-tug/GentiumPlus-I.ttf', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf','/usr/share/doc/texlive-doc/fonts/cm-unicode/Fontmap.CMU.otf','/usr/share/texlive/texmf-dist/fonts/opentype/public/inconsolata/Inconsolata.otf','/usr/share/fonts/truetype/inconsolata/Inconsolata.otf']
for j,i in enumerate(files_list):
        i = i.rstrip('\n')
	if i in excluding_list or "DejaVu" in i:
		continue
	print os.path.islink(i)
	print("real path is ", os.path.realpath(i))
        try:
	    argfontb=fontforge.open(os.path.realpath(i))
        except EnvironmentError:
	    print("it didnt work")
            #real path of i
            #do another try, except
            print("realpath = ",os.path.realpath(i))
	    argfontb=fontforge.open(os.path.realpath(i))
 
	glyphsare=argfontb.glyphs()
	unique_unicodes = {}
	cnt = 0
	for g in glyphsare:
	#	if g.unicode not in unique_unicodes:
	#		unique_unicodes[g.unicode] = 0
	#		continue
		#glyph_decimal_code = unique_unicodes[g.unicode]
                print("unicode = ",g.unicode)
                if g.unicode <0 or g.unicode>127:
                    continue
		intrinsic_glyph_decimal_code = g.unicode
                list_of_files=[]
		g.export(path + os.path.basename(i)+'-orig-'+ str(g.unicode) + '.bmp')
		g.transform(psMat.skew(.0523599)).export(path + os.path.basename(i)+'-skew-'+ str(g.unicode)+ '.bmp')
		list_of_files.append(path+os.path.basename(i) + '-orig-' + str(g.unicode)+ '.bmp')
		list_of_files.append(path+os.path.basename(i) + '-skew-' + str(g.unicode)+ '.bmp')

		g.transform(psMat.rotate(.0523599)).export(path + os.path.basename(i)+'-rotate-'+ str(g.unicode)+ '.bmp')
		list_of_files.append(path+os.path.basename(i) + '-rotate-' + str(g.unicode)+ '.bmp')

		g.transform(psMat.translate(1,1)).export(path + os.path.basename(i) + '-translate-' +str(g.unicode)+'.bmp')
		list_of_files.append(path+os.path.basename(i) + '-translate-' +str(g.unicode)+ '.bmp')

                
                for f in list_of_files:
			Image.open(f).convert('L').save("temporary_file.bmp")
			im = Image.open("temporary_file.bmp")
			im1=im.resize((28,28), Image.ANTIALIAS)
			imgArr=np.invert(np.array(im1.getdata(), np.uint8))
			imgArr.tofile(f)
		cnt+=1	
