import hashlib,struct,zlib,json
def digest(data):return hashlib.sha256(data).hexdigest()
def png(data):
 if data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('PNG signature')
 offset=8;compressed=[];dims=None;ended=False
 while offset<len(data):
  n=struct.unpack('>I',data[offset:offset+4])[0];kind=data[offset+4:offset+8];body=data[offset+8:offset+8+n];crc=struct.unpack('>I',data[offset+8+n:offset+12+n])[0]
  if len(body)!=n or zlib.crc32(kind+body)&0xffffffff!=crc:raise ValueError('PNG CRC')
  if kind==b'IHDR':dims=struct.unpack('>II',body[:8])
  if kind==b'IDAT':compressed.append(body)
  offset+=12+n
  if kind==b'IEND':ended=True;break
 if not ended or offset!=len(data) or dims!=(256,256):raise ValueError('PNG dimensions/end')
 if not zlib.decompress(b''.join(compressed)):raise ValueError('empty PNG pixels')

def expected(row):
 scene=row['name'].split('_context')[0];frames=row['frame_ids'];items=[]
 for sub in ('rgb','rgb_gt','depth','depth_gt'):
  items += [sub+'/'+scene+'_'+str(f)+'.png' for f in frames]
 for scope,ids in [('context',frames[:2]),('target',frames)]:
  for tag in ('pred','gt'):items += [scope+'_seg_'+tag+'/'+scene+'_'+tag+str(f)+'.png' for f in ids]
  items.append(scope+'_seg_pred/pred.json')
 return items

