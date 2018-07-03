from numpy import array, fromstring, uint8, reshape, ones
#-----------------------------------------------------------------------
# PNM stuff.

def noncomment(fd):
  """Read lines from the filehandle until a non-comment line is found. 
  Comments start with #"""
  while True:
    x = fd.readline() 
    if x.startswith('#') :
      continue
    else:
      return x

def writeLog(content):
  with open("./logfile.log", "a") as f:
    #print content
    f.write(content + "\n")     

def readPNM(fd):
  """Reads the PNM file from the filehandle"""
  t = noncomment(fd)  #P5
  s = noncomment(fd)  #3508 2481
  m = noncomment(fd) if not (t.startswith('P1') or t.startswith('P4')) else '1' #255

  #writeLog("In readPNM()\nt:{0}s:{1}m:{2}".format(t, s, m)) #t:P5 s:3400 4400 m:255 #table_down.pdf

  data = fd.read()
  ls = len(s.split())
  if ls != 2 :
    name = "<pipe>" if fd.name=="<fdopen>" else "Filename = {0}".format(fd.name)
    raise IOError("Expected 2 elements from parsing PNM file, got {0}: {1}".format(ls, name))
  xs, ys = s.split()
  width = int(xs)
  height = int(ys)
  m = int(m)

  if m != 255 :
    print "Just want 8 bit pgms for now!"
  
  d = fromstring(data,dtype=uint8)
  d = reshape(d, (height,width))
  return (m,width,height, d)
  #      (255, 3508, 2481, data)

def writePNM(fd,img):
  """Writes a PNM file to a filehandle given the img data as a numpy array"""
  s = img.shape
  m = 255
  if img.dtype == bool :
    img = img + uint8(0) 
    t = "P5"
    m = 1
  elif len(s) == 2 :
    t = "P5"
  else:
    t = "P6"
    
  fd.write( "%s\n%d %d\n%d\n" % (t, s[1],s[0],m) )
  fd.write( uint8(img).tostring() )


def dumpImage(outfile,bmp,img,bitmap=False, pad=2) :
    """Dumps the numpy array in image into the filename and closes the outfile"""
    oi = bmp if bitmap else img
    (height,width) = bmp.shape
    writePNM(outfile, oi[pad:height-pad, pad:width-pad])
    outfile.close()
