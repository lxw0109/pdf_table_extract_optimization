#coding: utf-8
import sys
import os
from numpy import array, fromstring, ones, zeros, uint8, diff, where, sum, delete, shape, argwhere #, concatenate
import subprocess
from .pnm import readPNM, dumpImage
import re
from pipes import quote
from xml.dom.minidom import getDOMImplementation
import json
import csv
import time

#-----------------------------------------------------------------------
def check_for_required_executable(name,command):
    """Checks for an executable called 'name' by running 'command' and supressing
    output. If the return code is non-zero or an OS error occurs, an Exception is raised"""
    try:
        with open(os.devnull, "w") as fnull:
            result=subprocess.check_call(command,stdout=fnull, stderr=fnull)
    except OSError as e:
        message = """Error running {0}.
Command failed: {1}
{2}""".format(name, " ".join(command), e)
        raise OSError(message)
    except subprocess.CalledProcessError as e:
        raise
    except Exception as e:
        raise

#-----------------------------------------------------------------------
def popen(name, command, *args, **kwargs):
    try:
        result = subprocess.Popen(command, *args, **kwargs)
        return result
    except OSError, e:
        message = """Error running {0}. Is it installed correctly?
Error: {1}""".format(name, e)
        raise OSError(message)
    except Exception, e:
        raise

def colinterp(a,x) :
    """Interpolates colors"""
    l = len(a)-1
    i = min(l, max(0, int (x * l)))
    (u,v) = a[i:i+2,:]
    return u - (u-v) * ((x * l) % 1.0)

colarr = array([ [255,0,0],[255,255,0],[0,255,0],[0,255,255],[0,0,255] ])

def col(x, colmult=1.0) :
    """colors"""
    return colinterp(colarr,(colmult * x)% 1.0) / 2

def writeLog(content):
  with open("./logfile.log", "a") as f:
    #print content
    f.write(content + "\n")

#hd: [332 333 448 450 557 558 666 667 774 775 915 916]  #table_down.pdf
#vd: [315  316  317  318  871  872 1424 1426 1979 1980 2533 2534 3087 3089] #table_down.pdf
def process_vd_hd(alist):
  """
  remove the cells that are much too small to contain any text.
  some lines are translated into two lines, which caused some empty rows and cols.
  """
  alist = alist.tolist()
  length = len(alist)
  if length < 4:
    return array(alist)

  span = 20
  index = 0
  delItem = []
  while index < length-2:
    if alist[index+2] - alist[index+1] < span:
      delItem.append(alist[index+1])
      delItem.append(alist[index+2])
    index += 2
  for item in delItem:
    alist.remove(item)

  alist[-2] -= 1  #600877_2015.pdf page9_table0 出现合并: 表格转为ppm图片后有些线变细，导致表格不闭合

  return array(alist)

def paintTable(bitmap_resolution, greyscale_threshold, pad, pg, infile, outfile, y, H):
  """
  paint the table into ppm
  pdftoppm:
    -x number
            Specifies the x-coordinate of the crop area top left corner
    -y number
        Specifies the y-coordinate of the crop area top left corner
    -W number
        Specifies the width of crop area in pixels (default is 0)
    -H number
        Specifies the height of crop area in pixels (default is 0)
  """
  p = popen("pdftoppm", ("pdftoppm -gray -r %d -f %d -l %d -y %d -H %d %s " %
      (bitmap_resolution, pg, pg, y, H, quote(infile))),
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
  (maxval, width, height, data) = readPNM(p.stdout)

  pad = int(pad)
  height += pad * 2
  width += pad * 2

  bmp = ones( (height,width) , dtype=bool )
  colorCritical = 100   #[92, 138] OK
  #bmp[pad:height-pad, pad:width-pad] = ( data[:,:] > int(255.0*greyscale_threshold/100.0) ) #greyscale_threshold: 25
  bmp[pad:height-pad, pad:width-pad] = ( data[:,:] > colorCritical )

  img = zeros( (height,width,3) , dtype=uint8 )
  img[:,:,0] = bmp*255
  img[:,:,1] = bmp*255
  img[:,:,2] = bmp*255

  dumpImage(outfile, bmp, img)

def splitTables(infile, pgs,
    specialFlag=False,  #lxw: add "specialFlag" to mark whether the table is special.
    outfilename=None,
    greyscale_threshold=25,
    page=None,
    #crop=None,
    line_length=0.17,
    bitmap_resolution=400,
    #name=None,
    pad=2,
    offset=10 #lxw: lthresh is too small, some Chinese word like "四" is considered to be a divider.
    #offset cannot be too large, because some tables are splited into two pages and the last row in the second page is very short(small height).
    ):
    #whitespace="normalize") :

  #writeLog("{0}\t\tIn core.py::splitTables()".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))))

  outfile = open(outfilename,'w') if outfilename else sys.stdout
  page = page or []
  (pg,frow,lrow) = (map(int,(pgs.split(":")))+[None,None])[0:3]

  p = popen("pdftoppm", ("pdftoppm -gray -r %d -f %d -l %d %s " %
      (bitmap_resolution, pg, pg, quote(infile))),
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

# image load secion.
  (maxval, width, height, data) = readPNM(p.stdout)

  pad = int(pad)
  height += pad * 2
  width += pad * 2

  bmp = ones( (height,width) , dtype=bool )
  #colorCritical = int(255.0*greyscale_threshold/100.0) #63
  colorCritical = 100   #[92, 138] OK
  #bmp[pad:height-pad, pad:width-pad] = ( data[:,:] > int(255.0*greyscale_threshold/100.0) ) #greyscale_threshold: 25
  bmp[pad:height-pad, pad:width-pad] = ( data[:,:] > colorCritical) #greyscale_threshold: 25

  """
  writeLog("colorCritical: {0}\n".format(colorCritical))
  #writeLog("data.shape:\n {0}\n".format(data.shape))
  #writeLog("data:\n {0}\n".format(data.tolist()))
  dataList = data.tolist()
  strList = []
  for eachRow in dataList:
    for item in eachRow:
      if item > 127 and item != 255:  #0: black, 255: white
        strList.append(str(item))
  writeLog("size: {0} data[:] > 0: \n {1}".format(len(strList), ",".join(strList)))
  """

# Set up Debuging image.
  img = zeros( (height,width,3) , dtype=uint8 )
  img[:,:,0] = bmp*255
  img[:,:,1] = bmp*255
  img[:,:,2] = bmp*255

  #dumpImage(outfile, bmp, img)

# Find bounding box.
  t = 0
  while t < height and sum(bmp[t,:]==0) == 0 :
    t=t+1
  if t > 0 :
    t=t-1

  b=height-1
  while b > t and sum(bmp[b,:]==0) == 0 :
    b=b-1
  if b < height-1:
    b = b+1

  l=0
  while l < width and sum(bmp[:,l]==0) == 0 :
    l=l+1
  if l > 0 :
    l=l-1

  r=width-1
  while r > l and sum(bmp[:,r]==0) == 0 :
    r=r-1
  if r < width-1 :
    r=r+1

# Line finding section.
# Find all vertical or horizontal lines that are more than rlthresh long, these are considered lines on the table grid.
  lthresh = int(line_length * bitmap_resolution) + offset #0.17 * 300 = 51; 0.17 * 400 = 68
  vs = zeros(width, dtype=int)
  for i in range(width) :
    dd = diff( where(bmp[:,i])[0] )
    if len(dd)>0:
      v = max ( dd )
      if v > lthresh : #最长的一段连续的黑色直线的长度大于lthresh(也就是表格中的分割线)
        vs[i] = 1
    else:
      # it was a solid black line.
      if bmp[0,i] == 0 : #这一列全是黑的
        vs[i] = 1
  vd = ( where(diff(vs[:]))[0] +1 ) #这一步经过where(diff)把vs中为1的下标(下标区间，如4,标出了4,5，详见numpyDemo.py中的whereDiff())，也就是把有竖线的列都标了出来

  if specialFlag:
    tempLthresh = lthresh
    lthresh = 1000   #magic number: 通常一整行的横线是2663,跨两列的这种是575,所以为了防止有些表的中间线段很长，取一个中间值1000
  hs = zeros(height, dtype=int)
  for j in range(height) :
    dd = diff( where(bmp[j,:])[0] )
    if len(dd) > 0 :
      h = max ( dd )
      #writeLog("h: {0}".format(h))
      if h > lthresh :
        hs[j] = 1
    else:
      # it was a solid black line.
      if bmp[j,0] == 0 :
        hs[j] = 1
        #writeLog("j: {0}".format(j))
        #writeLog("bmp[j,:]: {0}".format(bmp[j,:]))
  hd =( where(diff(hs[:]))[0] +1 ) #这一步经过where(diff)把hs中为1的下标(下标区间，如4,标出了4,5，详见numpyDemo.py中的whereDiff())，也就是把有横线的行都标了出来
  if specialFlag:
    lthresh = tempLthresh

  #vd = process_vd_hd(vd)  #remove the empty cell
  #hd = process_vd_hd(hd)  #remove the empty cell

  #writeLog("hd: {0}".format(hd))
  #writeLog("vd: {0}".format(vd))

# Look for dividers that are too large. #分割者；分隔器
  maxdiv=10
  #maxdiv=13 #some lines are very thick, 000407 P5, 10 is not enough
  i=0

  while i < len(vd) :
    if vd[i+1]-vd[i] > maxdiv :
      vd = delete(vd,i)
      vd = delete(vd,i)
    else:
      i=i+2

  j = 0
  while j < len(hd):
    if hd[j+1]-hd[j] > maxdiv :
      hd = delete(hd,j)
      hd = delete(hd,j)
    else:
      j=j+2

  #writeLog("hd: {0}".format(hd))
  #writeLog("vd: {0}".format(vd))


  #lxw: splitTables: split multiple tables in one page.
  def isDiv(a, l,r,t,b) :
    # if any col or row (in axis) is all zeros ...
    #writeLog("sum(bmp[t:b, l:r], axis=a): {0}".format( sum(sum(bmp[t:b, l:r], axis=a)==0) > 0 )) # l: 308, r: 3003, r-l: 2695, r-l+1???

    #lxw 有些表格画得不是很规范，会出现单元格不闭合的情况，此处对这种情况进行处理
    temp = 1
    if a == 0:  #col
        return sum( sum(bmp[t+temp:b-temp, l:r], axis=a)==0 ) > 0
    else:   #1, row
        return sum( sum(bmp[t:b, l+temp:r-temp], axis=a)==0 ) > 0
    #return sum( sum(bmp[t:b, l:r], axis=a)==0 ) > 0

  lenHd = len(hd)
  tables = [] #tuple of tables(tableStart, tableEnd)
  i = 0
  inTable = False
  while 2*(i+1) < lenHd-1:
    #ifDiv = isDiv(0, l+1, r, hd[2*i+1], hd[2*(i+1)])  #lxw NOTE: column "l" are all 0(bmp[:,l] = 0), l+1 is not good enough(when the most left pixel is the dividor of a table).
    ifDiv = isDiv(0, l, r+1, hd[2*i+1], hd[2*(i+1)]) #After deleting bmp[:,l] = 0, this is OK.
    if not ifDiv: #如果两条横线之间没有贯穿的竖线（即 不是分隔线divider)
      if inTable: #come out from a table
        tableEnd = hd[2*i+1]
        #writeLog("tableEnd: {0}".format(tableEnd))
        tables.append((tableStart, tableEnd))
        inTable = False
    else: # 如果两条横线之间有贯穿的竖线（即 是分隔线divider)
      if not inTable:
        inTable = True
        tableStart = hd[2*i]
    i += 1

  if inTable: #the last table
    tableEnd = hd[2*i+1]   # equals to: i -= 1; tableEnd = hd[2*(i+1)+1]
    tables.append((tableStart, tableEnd))

  #writeLog("\n{1}\ntables:\n {0}\n{1}".format(tables, "------"*10))

  newCells = []
  for tableStart, tableEnd in tables:
    length = len(infile)
    index = length -1
    while index >= 0:
      if infile[index] == "/":
        break
      index -= 1
    if index == -1: #no "/" in infile
      outputFile = infile[0:-4]
    else:
      outputFile = infile[index+1:-4]

    #tableFile = open("./output/{0}_{1}_{2}_{3}.ppm".format(outputFile, pg, tableStart, tableEnd),'w')
    #paintTable(bitmap_resolution, greyscale_threshold, pad, pg, infile, outfile, y, H):
    #paintTable(bitmap_resolution, greyscale_threshold, pad, pg, infile, tableFile, tableStart, tableEnd-tableStart)

    #process each table
    #def splitTables(infile, pgs, outfilename=None, greyscale_threshold=25, page=None, line_length=0.17, bitmap_resolution=400, pad=2):
    #newCells.append(process_page(infile, pgs, tableStart-3, tableEnd-tableStart+2, offset, checklines=True, outfilename="./output/{0}_{1}_{2}_{3}.ppm".format(outputFile, pg, tableStart, tableEnd)))

    if specialFlag:
      #split the special table(such as "number of shareholders" table) into lines.
      #newCells.append(process_page(infile, pgs, tableStart-3, tableEnd-tableStart+2, offset))
      newCells.append(processSpecial(hd, infile, pgs, tableStart, tableEnd, offset-offset, outputFile))
      #对于把表格逐行处理的情况，更加容易导致表格中比较矮的行被忽略掉，为了防止这种情况：在逐行处理时，把lthresh的值写得小一点儿(offset小一点儿)
      #in processSpecial(), we invoke process_page(infile, pgs, tableStart-3, tableEnd-tableStart+2, offset))
    else:
      #process_page(infile, pgs, y, H, offset)
      #newCells.append(process_page(infile, pgs, tableStart-3, tableEnd-tableStart+2, offset, outfilename="./output/{0}_{1}_{2}_{3}1.ppm".format(outputFile, pg, tableStart, tableEnd), checklines=True))
      newCells.append(process_page(infile, pgs, tableStart-3, tableEnd-tableStart+2, offset))

  #remove the empty table
  #writeLog("len(newCells): {0}".format(len(newCells)))
  #writeLog("type(newCells): {0}".format(type(newCells)))
  for cell in newCells:
    """
    if len(cell) == 0:  #remove empty table
      newCells.remove(cell)
      continue
    """
    #remove empty table & remove the table which contains only empty content
    emptyFlag = True
    for item in cell:
      if item[5] != '':
        emptyFlag = False
    if emptyFlag:
      newCells.remove(cell)

  return newCells

def processSpecial(hd, infile, pgs, tableStart, tableEnd, offset, outputFile):
  #process the special tables, such as "number of shareholders" table

  """
  writeLog("in processSpecial()\n----------------------------------------------------------------------")
  writeLog("hd: {0}".format(hd))
  writeLog("tableStart: {0}, tableEnd; {1}".format(tableStart, tableEnd))
  """
  startIndex = argwhere(hd==tableStart)
  startIndex = startIndex[0][0]
  #writeLog("startIndex: {0}".format(startIndex))

  endIndex = argwhere(hd==tableEnd)
  endIndex = endIndex[0][0]
  #writeLog("endIndex: {0}".format(endIndex))

  #startIndex = hd.index(tableStart)
  #endIndex = hd.index(tableEnd) + 1

  tableCells = []
  index = startIndex
  count = 0

  while index+1 != endIndex:
    """
    if count == 2 or count == 3:  # 有问题的两行
      tableFile = open("./output/{0}_{1}.ppm".format(hd[index]-3, hd[index+3]-hd[index]+2),'w')
      #paintTable(bitmap_resolution, greyscale_threshold, pad, pg, infile, outfile, y, H):
      paintTable(400, 25, 2, 33, infile, tableFile, hd[index]-3, hd[index+3]-hd[index]+2)
    """
    #process_page(infile, pgs, y, H, offset)
    #cells = process_page(infile, pgs, tableStart-3, tableEnd-tableStart+2, offset)
    cells = process_page(infile, pgs, hd[index]-3, hd[index+3]-hd[index]+2, offset) #cells: <type 'list'>
    #cells = process_page(infile, pgs, hd[index]-3, hd[index+3]-hd[index]+2, offset, outfilename="./output/{0}_{1}_{2}_{3}.ppm".format(outputFile, pgs, hd[index]-3, hd[index+3]-hd[index]+2)) #cells: <type 'list'>
    #cells是普通的python list和numpy数组没有关系
    #"cells" contains the content of one table(here one table is only one line)
    cells = [(col, row+count, colspan, rowspan, pg, content) for (col, row, colspan, rowspan, pg, content) in cells]
    count += 1
    count += getNumOfRows(cells) #cells[-1][1]  #cells中的元素（tuple），第二个单元（对应row）没有找到明显的规律，暂时不可用,所以目前还是调用了getNumOfRows()方法
    tableCells.extend(cells)
    #writeLog("tableCells: {0}\n\n".format(tableCells))
    index += 2

  return tableCells

def getNumOfRows(cells):
  """
  in processSpecial(): when process the special tables such as "number of shareholders" table, one row is parsed into
  more than multiple lines. we should count the number of lines.
  """
  #return cells[-1][1] #the last cell(index: -1) in cells, and the second element(index: 1, mean: row) in the tuple
  #cells中的元素（tuple），第二个单元（对应row）没有找到明显的规律，暂时不可用,所以目前还是调用了getNumOfRows()方法
  rowSet = []
  for cell in cells:
    for items in cells:
      rowSet.append(items[1])
  rowSet = set(rowSet)
  #writeLog("rowSet: {0}".format(rowSet))
  return len(rowSet) - 1


#cells = [pdf.process_page("./2015_601628.pdf", p) for p in pages]
#lxw NOTE: bitmap_resolution is a very important parameter, but it CANNOT be too large in case there are no (top,bottom,left,right) can be detected.
def process_page(infile, pgs, y, H, offset,
    outfilename=None,
    greyscale_threshold=25,
    page=None,
    crop=None,
    line_length=0.17,
    bitmap_resolution=400,
    name=None,
    pad=2,
    white=None,
    black=None,
    bitmap=False,
    checkcrop=False,
    checklines=False,
    checkdivs=False,
    checkcells=False,
    whitespace="normalize",
    boxes=False) :

  #writeLog("{0}\t\tIn core.py::process_page()".format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))))

  outfile = open(outfilename,'w') if outfilename else sys.stdout
  page = page or []
  (pg,frow,lrow) = (map(int,(pgs.split(":")))+[None,None])[0:3]
  #lxw INFO: pg, frow, lrow == 1, None, none
  #frow: first row?     lrow: last row?

  #check that pdftoppdm exists by running a simple command
  check_for_required_executable("pdftoppm",["pdftoppm","-h"])
  #end check

  p = popen("pdftoppm", ("pdftoppm -gray -r %d -f %d -l %d -y %d -H %d %s" %
      (bitmap_resolution, pg, pg, y, H, quote(infile))),
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)

  """
  p = popen("pdftoppm", ("pdftoppm -gray -r %d -f %d -l %d %s " %
      (bitmap_resolution, pg, pg, quote(infile))),
      stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
  #-gray  Generate a grayscale PGM file (instead of a color PPM file).
  #popen(name, command, *args, **kwargs):
  """

#-----------------------------------------------------------------------
# image load secion.
  #writeLog("type(p.stdout): {0}\np.stdout: {1}".format(type(p.stdout), p.stdout))
  #                                                      <type 'file'>   <open file '<fdopen>', mode 'rb' at 0x7fda95b29c90>

  (maxval, width, height, data) = readPNM(p.stdout)
  #(255,   3400,   4400, data)  #table_down.pdf
  #(255,   3400,   329, data)  #table_down.pdf #revoked in splitTables()
  #writeLog("width: {0}, height: {1}".format(width, height))

  pad = int(pad)
  height += pad * 2
  width += pad * 2

# reimbed image with a white padd.
  bmp = ones( (height,width) , dtype=bool )
  #bmp: (height+2) * (width+2)
  #   2 : heigth - 2, 2 : width - 2
  colorCritical = 100   #[92, 138] OK
  #bmp[pad:height-pad, pad:width-pad] = ( data[:,:] > int(255.0*greyscale_threshold/100.0) ) #greyscale_threshold: 25
  bmp[pad:height-pad, pad:width-pad] = ( data[:,:] > colorCritical)

  #writeLog("data.shape: {0}".format(data.shape)) #data.shape: (4400, 3400)  #table_down.pdf
  #writeLog("data:\n {0}\n".format(data))
  """
  data:
   [[255 255 255 ..., 255 255 255]
   [255 255 255 ..., 255 255 255]
   [255 255 255 ..., 255 255 255]
   ...,
   [255 255 255 ..., 255 255 255]
   [255 255 255 ..., 255 255 255]
   [255 255 255 ..., 255 255 255]]
  """
  #writeLog("bmp.shape: {0}".format(bmp.shape)) #bmp.shape: (4404, 3404)  #table_down.pdf
  #writeLog("bmp:\n {0}\n".format(list(bmp))) #All elements are True here.
  """
  bmp:
   [[ True  True  True ...,  True  True  True]
   [ True  True  True ...,  True  True  True]
   [ True  True  True ...,  True  True  True]
   ...,
   [ True  True  True ...,  True  True  True]
   [ True  True  True ...,  True  True  True]
   [ True  True  True ...,  True  True  True]]
  """

# Set up Debuging image.
  img = zeros( (height,width,3) , dtype=uint8 )
  img[:,:,0] = bmp*255
  img[:,:,1] = bmp*255
  img[:,:,2] = bmp*255

  #dumpImage(outfile,bmp,img)

  #writeLog("img.shape: {0}".format(img.shape)) #img.shape: (4404, 3404, 3) #table_down.pdf
  #writeLog("img:\n {0}\n".format(img))
  """
  img:
  [
   [[255 255 255]
    [255 255 255]
    [255 255 255]
    ...,
    [255 255 255]
    [255 255 255]
    [255 255 255]]

   [[255 255 255]
    [255 255 255]
    [255 255 255]
    ...,
    [255 255 255]
    [255 255 255]
    [255 255 255]]

   [[255 255 255]
    [255 255 255]
    [255 255 255]
    ...,
    [255 255 255]
    [255 255 255]
    [255 255 255]]

   ...,
   [[255 255 255]
    [255 255 255]
    [255 255 255]
    ...,
    [255 255 255]
    [255 255 255]
    [255 255 255]]

   [[255 255 255]
    [255 255 255]
    [255 255 255]
    ...,
    [255 255 255]
    [255 255 255]
    [255 255 255]]

   [[255 255 255]
    [255 255 255]
    [255 255 255]
    ...,
    [255 255 255]
    [255 255 255]
    [255 255 255]]
  ]
  """

#-----------------------------------------------------------------------
  """
  data.shape: (4400, 3400) value: 255 #table_down.pdf
  bmp.shape: (4404, 3404) value: True #table_down.pdf
  img.shape: (4404, 3404, 3) value: 255 #table_down.pdf
  """
# Find bounding box.
  t = 0
  while t < height and sum(bmp[t,:]==0) == 0 :
  #当bmp中存在值为False（==0）的元素时,停止，表示找到了top. sum(expression)加的是expression的结果，一个变量也是一个表达式
    t=t+1
  if t > 0 :
    t=t-1

  b=height-1
  while b > t and sum(bmp[b,:]==0) == 0 :
    b=b-1
  if b < height-1:
    b = b+1

  l=0
  while l < width and sum(bmp[:,l]==0) == 0 :
    l=l+1
  if l > 0 :
    l=l-1

  r=width-1
  while r > l and sum(bmp[:,r]==0) == 0 :
    r=r-1
  if r < width-1 :
    r=r+1

  #table_down.pdf
  #4400 * 3400
  #writeLog("lxw t: {0}\tb: {1}\tl: {2}\tr:{3}".format(t, b, l, r))
  #t: 332  b: 915  l: 315  r:3088
  #t: 2  b: 330  l: 315  r:3088  #revoked in splitTables()
  #writeLog("bmp:\n {0}\n".format(list(bmp))) #All elements are True here.

# Mark bounding box.
  """
  #20161026
  #lxw NOTE: comment here, because a lot of problems are caused by this.
  bmp[t,:] = 0
  bmp[b,:] = 0
  bmp[:,l] = 0
  bmp[:,r] = 0
  """
  #writeLog("bmp:\n {0}\n".format(list(bmp)))
  #writeLog("bmp[t,:]:\n{0}\n\nbmp[b,:]:\n{1}\n\nbmp[:,l]:\n{2}\n\nbmp[:,r]:\n{3}\n".format(list(bmp[t,:]), list(bmp[b,:]), list(bmp[:,l]), list(bmp[:,r])))

  def boxOfString(x,p) :
    s = x.split(":")
    if len(s) < 4 :
      raise ValueError("boxes have format left:top:right:bottom[:page]")
    return ([bitmap_resolution * float(x) + pad for x in s[0:4] ]
                + [ p if len(s)<5 else int(s[4]) ] )
  """
  outfilename=None, greyscale_threshold=25, page=None, crop=None, line_length=0.17, bitmap_resolution=300, name=None,
  pad=2, white=None, black=None, bitmap=False, checkcrop=False, checklines=False, checkdivs=False, checkcells=False,
  whitespace="normalize", boxes=False
  """
# translate crop to paint white.
  whites = []
  if crop :
    (l,t,r,b,p) = boxOfString(crop,pg)
    whites.extend( [ (0,0,l,height,p), (0,0,width,t,p),
                     (r,0,width,height,p), (0,b,width,height,p) ] )

  #cells: [[(col, row, colspan, rowspan, page, "content of the cell"), (col, row, colspan, rowspan, page, "content of the cell"),..., (col, row, colspan, rowspan, page, "content of the cell")]]

# paint white ...
  if white :
    whites.extend( [ boxOfString(b, pg) for b in white ] )

  for (l,t,r,b,p) in whites :
    if p == pg :
      bmp[ t:b+1,l:r+1 ] = 1
      img[ t:b+1,l:r+1 ] = [255,255,255]

# paint black ...
  if black :
    for b in black :
      (l,t,r,b) = [bitmap_resolution * float(x) + pad for x in b.split(":") ]
      bmp[ t:b+1,l:r+1 ] = 0
      img[ t:b+1,l:r+1 ] = [0,0,0]

  if checkcrop :
    dumpImage(outfile,bmp,img, bitmap, pad)
    return True

  #writeLog("bmp[t,:]:\n{0}\n\nbmp[b,:]:\n{1}\n\nbmp[:,l]:\n{2}\n\nbmp[:,r]:\n{3}\n".format(list(bmp[t,:]), list(bmp[b,:]), list(bmp[:,l]), list(bmp[:,r])))
#-----------------------------------------------------------------------
# Line finding section.
#
# Find all vertical or horizontal lines that are more than rlthresh
# long, these are considered lines on the table grid.

  lthresh = int(line_length * bitmap_resolution) + offset #0.17 * 300 = 51; 0.17 * 400 = 68
  vs = zeros(width, dtype=int)
  for i in range(width) :
    dd = diff( where(bmp[:,i])[0] )
    """
    where(bmp[:,i])[0]: 找出第i列 值为True的元素的下标(黑色的是False，白色的是True)
    diff( where(bmp[:,i])[0] ) : 计算相邻元素的差值
    """
    if len(dd)>0:
      v = max ( dd )
      if v > lthresh : #最长的一段连续的黑色直线的长度大于lthresh(也就是表格中的分割线)
        vs[i] = 1
    else:
      # it was a solid black line.
      if bmp[0,i] == 0 : #这一列全是黑的
        vs[i] = 1
  vd = ( where(diff(vs[:]))[0] +1 ) #这一步经过where(diff)把vs中为1的下标(下标区间，如4,标出了4,5，详见numpyDemo.py中的whereDiff())，也就是把有竖线的列都标了出来
  #writeLog("vs:\n{0}".format(vs))
  #vs: [0 0 0 ..., 0 0 0]
  #writeLog("diff(vs[:]):\n{0}".format(diff(vs[:])))
  #diff(vs[:]): [0 0 0 ..., 0 0 0]
  #writeLog("where(diff(vs[:])):\n{0}".format(where(diff(vs[:]))))
  #where(diff(vs[:])): (array([ 314,  315,  316,  317,  870,  871, 1423, 1425, 1978, 1979, 2532, 2533, 3086, 3088]),) #table_down.pdf
  #writeLog("vd:\n{0}".format(vd))
  #vd: [315  316  317  318  871  872 1424 1426 1979 1980 2533 2534 3087 3089] #table_down.pdf

  hs = zeros(height, dtype=int)
  for j in range(height) :
    dd = diff( where(bmp[j,:])[0] )
    if len(dd) > 0 :
      h = max ( dd )
      if h > lthresh :
        hs[j] = 1
    else:
      # it was a solid black line.
      if bmp[j,0] == 0 :
        hs[j] = 1
        #writeLog("j: {0}".format(j))
        #writeLog("bmp[j,:]: {0}".format(bmp[j,:]))
  hd =( where(diff(hs[:]))[0] +1 ) #这一步经过where(diff)把hs中为1的下标(下标区间，如4,标出了4,5，详见numpyDemo.py中的whereDiff())，也就是把有横线的行都标了出来
  #writeLog("hd:\n{0}".format(hd))
  #hd: [332 333 448 450 557 558 666 667 774 775 915 916]  #table_down.pdf

  #writeLog("shape(vd):\n{0}".format(shape(vd)))
  #writeLog("shape(hd):\n{0}".format(shape(hd)))
  #writeLog('---'*20)
  vd = process_vd_hd(vd)  #remove the empty cell
  hd = process_vd_hd(hd)  #remove the empty cell
  #writeLog("vd:\n{0}".format(vd))
  #writeLog("hd:\n{0}".format(hd))
  #writeLog("shape(vd):\n{0}".format(vd.shape))
  #writeLog("shape(hd):\n{0}".format(hd.shape))
  #exit(0)

#-----------------------------------------------------------------------
  """
  outfilename=None, greyscale_threshold=25, page=None, crop=None, line_length=0.17, bitmap_resolution=300, name=None, pad=2, white=None,
  black=None, bitmap=False, checkcrop=False, checklines=False, checkdivs=False, checkcells=False, whitespace="normalize", boxes=False
  """
# Look for dividers that are too large. #分割者；分隔器
  #maxdiv=10
  maxdiv=25
  #maxdiv=25000000
  i=0

  while i < len(vd) :
    if vd[i+1]-vd[i] > maxdiv :
      vd = delete(vd,i)
      vd = delete(vd,i)
    else:
      i=i+2

  j = 0
  while j < len(hd):
    if hd[j+1]-hd[j] > maxdiv :
      hd = delete(hd,j)
      hd = delete(hd,j)
    else:
      j=j+2

  #writeLog("vd:\n{0}".format(vd))
  #writeLog("hd:\n{0}".format(hd))

  def isDiv(a, l,r,t,b) :
    #writeLog("in isDiv(): l: {0}, r: {1}, t: {2}, b: {3}".format(l, r, t, b))
    # if any col or row (in axis) is all zeros ...
    #writeLog("sum(bmp[t:b, l:r], axis=a): {0}".format( sum(sum(bmp[t:b, l:r], axis=a)==0) > 0 )) # l: 308, r: 3003, r-l: 2695, r-l+1???

    #lxw 有些表格画得不是很规范，会出现单元格不闭合的情况，此处对这种情况进行处理
    temp = 1
    if a == 0:  #col
        return sum( sum(bmp[t+temp:b-temp, l:r], axis=a)==0 ) > 0
    else:   #1, row
        return sum( sum(bmp[t:b, l+temp:r-temp], axis=a)==0 ) > 0
    #return sum( sum(bmp[t:b, l:r], axis=a)==0 ) > 0

  """
  #lxw: splitTables: split multiple tables in one page.
  lenHd = len(hd)
  tables = [] #tuple of tables(tableStart, tableEnd)
  i = 0
  inTable = False
  while 2*(i+1) < lenHd-1:
    #ifDiv = isDiv(0, l+1, r, hd[2*i+1], hd[2*(i+1)])  #lxw NOTE: column "l" are all 0(bmp[:,l] = 0), l+1 is not good enough(when the most left pixel is the dividor of a table).
    ifDiv = isDiv(0, l, r+1, hd[2*i+1], hd[2*(i+1)]) #After deleting bmp[:,l] = 0, this is OK.
    if not ifDiv: #如果两条横线之间没有贯穿的竖线（即 不是分隔线divider)
      if inTable: #come out from a table
        tableEnd = hd[2*i+1]
        #writeLog("tableEnd: {0}".format(tableEnd))
        tables.append((tableStart, tableEnd))
        inTable = False
    else: # 如果两条横线之间有贯穿的竖线（即 是分隔线divider)
      if not inTable:
        inTable = True
        tableStart = hd[2*i]
    i += 1

  if inTable: #the last table
    tableEnd = hd[2*i+1]   # equals to: i -= 1; tableEnd = hd[2*(i+1)+1]
    tables.append((tableStart, tableEnd))

  writeLog("\n{1}\ntables:\n {0}\n{1}".format(tables, "------"*10))

  #paintTable(bitmap_resolution, greyscale_threshold, pad, pg, infile, outfile, y, H):
  for tableStart, tableEnd in tables:

    length = len(infile)
    index = length -1
    while index >= 0:
      if infile[index] == "/":
        break
      index -= 1
    if index != -1: #no "/" in infile
      outputFile = infile[0:-4]
    else:
      outputFile = infile[index+1:-4]

    tableFile = open("./output/{0}_{1}_{2}.ppm".format(outputFile, tableStart, tableEnd),'w')
    paintTable(bitmap_resolution, greyscale_threshold, pad, pg, infile, tableFile, tableStart-3, tableEnd-tableStart+2)
  """

  if checklines :
    for i in vd :
      img[:,i] = [255,0,0] # red

    for j in hd :
      img[j,:] = [0,0,255] # blue
    dumpImage(outfile,bmp,img)
    return True

  #writeLog("After deleting the dividers that are too large:")
  #writeLog("vd:\n{0}".format(vd))
  #vd: [ 315  316  317  318  871  872 1424 1426 1979 1980 2533 2534 3087 3089] #table_down.pdf
  #writeLog("hd:\n{0}".format(hd))
  #hd: [332 333 448 450 557 558 666 667 774 775 915 916]  #table_down.pdf
  #lxw NOTE: almost no changing after deleting the dividers that are too large.

#-----------------------------------------------------------------------
# divider checking.
#
# at this point vd holds the x coordinate of vertical  and
# hd holds the y coordinate of horizontal divider tansitions for each
# vertical and horizontal lines in the table grid.

  """
  def isDiv(a, l,r,t,b) :
    # if any col or row (in axis) is all zeros ...
    return sum( sum(bmp[t:b, l:r], axis=a)==0 ) >0
  """

  if checkdivs :
    img = img / 2 #255 / 2 == 127
    for j in range(0,len(hd),2):
      for i in range(0,len(vd),2):
        if i > 0 :
          (l,r,t,b) = (vd[i-1], vd[i],   hd[j],   hd[j+1])
          img[ t:b, l:r, 1 ] = 192
          if isDiv(1, l,r,t,b) :
            img[ t:b, l:r, 0 ] = 0
            #writeLog("A: img[ t:b, l:r, 2 ]: {0}".format(img[ t:b, l:r, 2 ]))   #127
            img[ t:b, l:r, 2 ] = 255  #rgb(0, 192, 255) 蓝色      #rgb(0, 192, 127) 绿色

        if j > 0 :
          (l,r,t,b) = (vd[i],   vd[i+1], hd[j-1], hd[j] )
          img[ t:b, l:r, 1 ] = 128
          if isDiv(0, l,r,t,b) :
            img[ t:b, l:r, 0 ] = 255
            #writeLog("B: img[ t:b, l:r, 2 ]: {0}".format(img[ t:b, l:r, 2 ]))  #127
            img[ t:b, l:r, 2 ] = 0  #rgb(255, 128, 0) 橘黄        #rgb(255, 128, 127) 粉色
    dumpImage(outfile,bmp,img)
    return True

#-----------------------------------------------------------------------
# Cell finding section.
# This algorithum is width hungry, and always generates rectangular
# boxes.

  #cells: [[(col, row, colspan, rowspan, page, "content of the cell"), (col, row, colspan, rowspan, page, "content of the cell"),..., (col, row, colspan, rowspan, page, "content of the cell")]]
  #hd: [332 333 448 450 557 558 666 667 774 775 915 916]  #table_down.pdf
  #vd: [315 316 317 318 871 872 1424 1426 1979 1980 2533 2534 3087 3089] #table_down.pdf
  cells =[]
  touched = zeros( (len(hd), len(vd)), dtype=bool )
  #writeLog("len(hd)):\n {0}\n".format(len(hd))) #12 table_down.pdf
  #writeLog("len(vd)):\n {0}\n".format(len(vd))) #14 table_down.pdf

  #lxw NOTE:着重看一下，vd和hd是如何处理的, vd和hd是什么关系？

  j = 0
  while j*2+2 < len(hd) :
    i = 0
    while i*2+2 < len(vd) :
      u = 1
      v = 1
      if not touched[j,i] :
        '''
        writeLog("2+(i+u)*2:{0}\tlen(vd):{1}\t2+(i+u)*2 < len(vd):{2}".format(2+(i+u)*2, len(vd), 2+(i+u)*2 < len(vd)))
        writeLog("2*(i+u): {0}, vd[ 2*(i+u) ]: {1}\n2*(i+u)+1: {2}, vd[ 2*(i+u)+1]: {3}\n2*(j+v)-1: {4}, hd[ 2*(j+v)-1 ]: {5}\n2*(j+v): {6}, hd[ 2*(j+v) ]: {7}"\
          .format(2*(i+u), vd[ 2*(i+u) ], 2*(i+u)+1, vd[ 2*(i+u)+1], 2*(j+v)-1, hd[ 2*(j+v)-1 ], 2*(j+v), hd[ 2*(j+v) ]))
        writeLog("isDiv( 0, vd[ 2*(i+u) ], vd[ 2*(i+u)+1], hd[ 2*(j+v)-1 ], hd[ 2*(j+v) ] ): {0}".format(isDiv( 0, vd[ 2*(i+u) ], vd[ 2*(i+u)+1], hd[ 2*(j+v)-1 ], hd[ 2*(j+v) ])) )
        '''
        while 2+(i+u)*2 < len(vd) and not isDiv( 0, vd[ 2*(i+u) ], vd[ 2*(i+u)+1], hd[ 2*(j+v)-1 ], hd[ 2*(j+v) ] ):
          '''
          writeLog("2+(i+u)*2:{0}\tlen(vd):{1}\t2+(i+u)*2 < len(vd):{2}".format(2+(i+u)*2, len(vd), 2+(i+u)*2 < len(vd)))
          writeLog("2*(i+u): {0}, vd[ 2*(i+u) ]: {1}\n2*(i+u)+1: {2}, vd[ 2*(i+u)+1]: {3}\n2*(j+v)-1: {4}, hd[ 2*(j+v)-1 ]: {5}\n2*(j+v): {6}, hd[ 2*(j+v) ]: {7}"\
            .format(2*(i+u), vd[ 2*(i+u) ], 2*(i+u)+1, vd[ 2*(i+u)+1], 2*(j+v)-1, hd[ 2*(j+v)-1 ], 2*(j+v), hd[ 2*(j+v) ]))
          writeLog("A: isDiv( 0, vd[ 2*(i+u) ], vd[ 2*(i+u)+1], hd[ 2*(j+v)-1 ], hd[ 2*(j+v) ] ): {0}".format(isDiv( 0, vd[ 2*(i+u) ], vd[ 2*(i+u)+1], hd[ 2*(j+v)-1 ], hd[ 2*(j+v) ])) )
          '''
          u=u+1

        bot = False
        '''
        writeLog("2+(j+v)*2:{0}\tlen(hd):{1}\t2+(j+v)*2 < len(hd):{2}".format(2+(j+v)*2, len(hd), 2+(j+v)*2 < len(hd)))
        '''
        while 2+(j+v)*2 < len(hd) and not bot :
          '''
          writeLog("2+(j+v)*2:{0}\tlen(hd):{1}\t2+(j+v)*2 < len(hd):{2}".format(2+(j+v)*2, len(hd), 2+(j+v)*2 < len(hd)))
          '''
          bot = False
          for k in range(1, u+1) :
            '''
            writeLog("2*(i+k)-1: {0}, vd[2*(i+k)-1]: {1}\n2*(i+k): {2}, vd[2*(i+k)]: {3}\n2*(j+v): {4}, hd[2*(j+v)]: {5}\n2*(j+v)+1: {6}, hd[ 2*(j+v)+1 ]: {7}"\
              .format(2*(i+k)-1, vd[2*(i+k)-1], 2*(i+k),  vd[2*(i+k)], 2*(j+v), hd[2*(j+v)], 2*(j+v)+1, hd[ 2*(j+v)+1 ]))
            writeLog("B: isDiv( 1, vd[ 2*(i+k)-1 ], vd[ 2*(i+k)], hd[ 2*(j+v) ], hd[ 2*(j+v)+1 ] ): {0}".format(isDiv( 1, vd[ 2*(i+k)-1 ], vd[ 2*(i+k)], hd[ 2*(j+v) ], hd[ 2*(j+v)+1 ] )))
            '''
            bot |= isDiv( 1, vd[ 2*(i+k)-1 ], vd[ 2*(i+k)], hd[ 2*(j+v) ], hd[ 2*(j+v)+1 ] )
          if not bot :
            v=v+1
        cells.append( (i,j,u,v) )

        #writeLog("i, j, u, v: {0}, {1}, {2}, {3}\n".format(i, j, u, v))
        #i: col;  j: row;   u: colspan;  v: rowspan;
        touched[ j:j+v, i:i+u] = True
      i = i + 1
    j = j + 1

  if checkcells :
    nc = len(cells)+0.
    img = img / 2
    for k in range(len(cells)):
      (i,j,u,v) = cells[k]
      (l,r,t,b) = ( vd[2*i+1] , vd[ 2*(i+u) ], hd[2*j+1], hd[2*(j+v)] )
      #img[ t:b, l:r ] += col( k/nc )  #Error:TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('uint8') with casting rule 'same_kind'

      #lxw
      fNum = col(k/nc)
      #writeLog("col(k/nc):\n {0}\n".format(fNum))
      fNum = fNum.astype(uint8)
      #writeLog("After astype: col(k/nc):\n {0}\n".format(fNum)) #标量的；数量的
      img[ t:b, l:r ] += fNum

    dumpImage(outfile,bmp,img)
    return True

#-----------------------------------------------------------------------
# fork out to extract text for each cell.

  whitespace = re.compile( r'\s+')

  def getCell( (i,j,u,v) ):
    (l,r,t,b) = ( vd[2*i+1] , vd[ 2*(i+u) ], hd[2*j+1], hd[2*(j+v)] )
    #writeLog("l, r, t, b: {0}, {1}, {2}, {3}\n".format(l, r, t, b))

    p = popen("pdftotext",
              "pdftotext -r %d -x %d -y %d -W %d -H %d -layout -nopgbrk -f %d -l %d %s -" %
              #(bitmap_resolution, l-pad, t-pad+2400, r-l, b-t, pg, pg, quote(infile)),
              (bitmap_resolution, l-pad, t-pad+y, r-l, b-t, pg, pg, quote(infile)),
              stdout=subprocess.PIPE,
              shell=True )
    """
       -r number
              Specifies the resolution, in DPI.  The default is 72 DPI.

       -x number
              Specifies the x-coordinate of the crop area top left corner

       -y number
              Specifies the y-coordinate of the crop area top left corner

       -W number
              Specifies the width of crop area in pixels (default is 0)

       -H number
              Specifies the height of crop area in pixels (default is 0)

       -layout
              Maintain (as best as possible) the original physical layout of the
              text. The  default is to 'undo' physical layout (columns, hyphen‐
              ation, etc.) and output the text in reading order. hyphenation: 连字符

       -nopgbrk
              Don't insert page breaks (form feed characters, 换页字符) between pages.

       -f number
              Specifies the first page to convert.

       -l number
              Specifies the last page to convert.
    """

    ret = p.communicate()[0]
    #writeLog("ret: {0}".format(ret)) #lxw: "ret" is the content of the cell

    if whitespace != 'raw' :
      #writeLog("whitespace: {0}\n".format(str(whitespace)))
      ret = whitespace.sub( "" if whitespace == "none" else " ", ret )
      #writeLog("ret: {0}\n".format(ret))  #lxw: "ret" is the content of the cell
      if len(ret) > 0 :
        ret = ret[ (1 if ret[0]==' ' else 0) :
                   len(ret) - (1 if ret[-1]==' ' else 0) ]  #lxw: remove the beginning and trailing " "(space)
        #writeLog("ret: {0}\n".format(ret))  #lxw: "ret" is the content of the cell
    return (i,j,u,v,pg,ret)

  if boxes :
    cells = [ x + (pg,"",) for x in cells if ( frow == None or (x[1] >= frow and x[1] <= lrow)) ]
  else :
    #check that pdftotext exists by running a simple command
    check_for_required_executable("pdftotext", ["pdftotext","-h"])
    #end check
    cells = [ getCell(x) for x in cells if ( frow == None or (x[1] >= frow and x[1] <= lrow)) ]
    #lxw: frow: first row     lrow: last row

    #writeLog("cells: {0}\n".format(cells))
    """
    cells: [
    (0, 0, 6, 1, 1, 'Above table.'),
    (0, 1, 1, 1, 1, ''), (1, 1, 1, 1, 1, '1'), (2, 1, 1, 1, 1, '23'), (3, 1, 1, 1, 1, '4'), (4, 1, 1, 1, 1, '5'), (5, 1, 1, 1, 1, '6789'),
    (0, 2, 1, 1, 1, ''), (1, 2, 1, 1, 1, '10'), (2, 2, 1, 1, 1, '11'), (3, 2, 1, 1, 1, '12'), (4, 2, 1, 1, 1, '13'), (5, 2, 1, 1, 1, '14 15'),
    (0, 3, 1, 2, 1, ''), (1, 3, 1, 1, 1, '16'), (2, 3, 1, 1, 1, '17 18 19'), (3, 3, 1, 1, 1, '20 21'), (4, 3, 1, 1, 1, '22 23 24'), (5, 3, 1, 1, 1, '25'),
    (1, 4, 5, 1, 1, 'Below table.')
    ]
    """
  return cells

#-----------------------------------------------------------------------
#output section.

def output(cells, pgs,
                cells_csv_filename=None,
                cells_json_filename=None,
                cells_xml_filename=None,
                table_csv_filename=None,
                table_html_filename=None,
                table_list_filename=None,
                infile=None, name=None, output_type=None
                ):

    output_types = [
             dict(filename=cells_csv_filename, function=o_cells_csv),
             dict(filename=cells_json_filename, function=o_cells_json),
             dict(filename=cells_xml_filename, function=o_cells_xml),
             dict(filename=table_csv_filename, function=o_table_csv),
             dict(filename=table_html_filename, function=o_table_html),
             dict(filename=table_list_filename, function=o_table_list)
             ]

    for entry in output_types:
        if entry["filename"]:
            if entry["filename"] != sys.stdout:
                outfile = open(entry["filename"],'w')
            else:
                outfile = sys.stdout

            entry["function"](cells, pgs,
                                outfile=outfile,
                                name=name,
                                infile=infile,
                                output_type=output_type)

            if entry["filename"] != sys.stdout:
                outfile.close()

def o_cells_csv(cells,pgs, outfile=None, name=None, infile=None, output_type=None) :
  outfile = outfile or sys.stdout
  csv.writer( outfile , dialect='excel' ).writerows(cells)

def o_cells_json(cells,pgs, outfile=None, infile=None, name=None, output_type=None) :
  """Output JSON formatted cell data"""
  outfile = outfile or sys.stdout
  #defaults
  infile=infile or ""
  name=name or ""

  json.dump({
    "src": infile,
    "name": name,
    "colnames": ( "x","y","width","height","page","contents" ),
    "cells":cells
    }, outfile)

def o_cells_xml(cells,pgs, outfile=None,infile=None, name=None, output_type=None) :
  """Output XML formatted cell data"""
  outfile = outfile or sys.stdout
  #defaults
  infile=infile or ""
  name=name or ""

  doc = getDOMImplementation().createDocument(None,"table", None)
  root = doc.documentElement;
  if infile :
    root.setAttribute("src",infile)
  if name :
    root.setAttribute("name",name)
  for cl in cells :
    x = doc.createElement("cell")
    map(lambda(a): x.setAttribute(*a), zip("xywhp",map(str,cl)))
    if cl[5] != "" :
      x.appendChild( doc.createTextNode(cl[5]) )
    root.appendChild(x)
  outfile.write( doc.toprettyxml() )

def table_to_list(cells,pgs) :
  """Output list of lists"""
  l=[0,0,0]
  for (i,j,u,v,pg,value) in cells :
      r=[i,j,pg]
      l = [max(x) for x in zip(l,r)]

  tab = [ [ [ "" for x in range(l[0]+1)
            ] for x in range(l[1]+1)
          ] for x in range(l[2]+1)
        ]
  for (i,j,u,v,pg,value) in cells :
    tab[pg][j][i] = value

  return tab

def o_table_csv(cells,pgs, outfile=None, name=None, infile=None, output_type=None) :
  """Output CSV formatted table"""
  outfile = outfile or sys.stdout
  tab=table_to_list(cells, pgs)
  for t in tab:
    csv.writer( outfile , dialect='excel' ).writerows(t)


def o_table_list(cells,pgs, outfile=None, name=None, infile=None, output_type=None) :
  """Output list of lists"""
  outfile = outfile or sys.stdout
  tab = table_to_list(cells, pgs)
  print(tab)

def o_table_html(cells,pgs, outfile=None, output_type=None, name=None, infile=None) :
  """Output HTML formatted table"""

  oj = 0
  opg = 0
  doc = getDOMImplementation().createDocument(None,"table", None)
  root = doc.documentElement;
  if (output_type == "table_chtml" ):
    root.setAttribute("border","1")
    root.setAttribute("cellspaceing","0")
    root.setAttribute("style","border-spacing:0")
  nc = len(cells)
  tr = None
  for k in range(nc):
    (i,j,u,v,pg,value) = cells[k]
    if j > oj or pg > opg:
      if pg > opg:
        s = "Name: " + name + ", " if name else ""
        root.appendChild( doc.createComment( s +
          ("Source: %s page %d." % (infile, pg) )));
      if tr :
        root.appendChild(tr)
      tr = doc.createElement("tr")
      oj = j
      opg = pg
    td = doc.createElement("td")
    if value != "" :
      td.appendChild( doc.createTextNode(value) )
    if u>1 :
      td.setAttribute("colspan",str(u))
    if v>1 :
      td.setAttribute("rowspan",str(v))
    if output_type == "table_chtml" :
      td.setAttribute("style", "background-color: #%02x%02x%02x" %
            tuple(128+col(k/(nc+0.))))
    tr.appendChild(td)
  root.appendChild(tr)
  outfile.write( doc.toprettyxml() )
