import os
import re
from percolation import c


here = os.path.abspath(os.path.dirname(__file__))
ECANTORIXDIR = here + '/ecantorix'
ECANTORIXCACHE = ECANTORIXDIR + '/cache'
if not os.path.isdir(ECANTORIXCACHE):
    os.mkdir(ECANTORIXCACHE)


def sing(text="ma-ry had a lit-tle lamb",
        notes=(4,2,0,2,4,4,4),durs=(1,1,1,1,1,1,2), reference=60):
    # write abc file
    # write make file
    # convert file to midi
    # sing it out
    writeAbc(text,notes,durs)
    # os.system('{}/make'.format(ECANTORIXCACHE))

    
def writeAbc(text,notes,durs,M='4/4',L='1/4',Q=120,K='C',reference=60):
    text_='X:1\n'
    text_+='T:Some chanting for music python package\n'
    text_+='M:{}\n'.format(M)
    text_+='L:{}\n'.format(L)
    text_+='Q:{}\n'.format(Q)
    text_+='V:1\n'
    text_+='K:{}\n'.format(K)
    notes = translateToAbc(notes, durs, reference)
    text_ += notes + "\nw: " + text
    fname = ECANTORIXCACHE + "/achant.abc"
    with open(fname,'w') as f:
        f.write(text_)

def translateToAbc(notes, durs, reference):
    durs = [str(i).replace('-','/') for i in durs]
    durs = [i if i != '1' else '' for i in durs]
    notes = converter.convert(notes, reference)
    return ''.join([i+j for i, j in zip(notes,durs)])

    
class Notes:
    def makeDict(self):
        notes=re.findall(r'[\^]{0,1}[a-g]{1}','c^cd^def^fg^ga^ab')
        self.notes=re.findall(r'[\^]{0,1}[a-g]{1}','c^cd^def^fg^ga^ab')
        notes_=[note.upper() for note in notes]
        notes__=[note+"," for note in notes_]
        notes___=[note+"," for note in notes__]
        notes_u=[note+"'" for note in notes]
        notes__u=[note+"'" for note in notes_u]
        notes___u=[note+"'" for note in notes__u]
        notes_all = notes___+notes__+notes_+notes+notes_u+notes__u+notes___u
        self.notes_dict = dict([(i,j) for i, j in zip(range(24,97),notes_all)])
        self.notes_=[note.upper() for note in notes]
        self.notes__=[note+"," for note in notes_]
        self.notes___=[note+"," for note in notes__]
        self.notes_u=[note+"'" for note in notes]
        self.notes__u=[note+"'" for note in notes_u]
        self.notes___u=[note+"'" for note in notes__u]
        self.notes_all = notes___+notes__+notes_+notes+notes_u+notes__u+notes___u

    def convert(self, notes, reference):
        if 'notes_dict' not in dir(self):
            self.makeDict()
        notes_ = [reference + note for note in notes]
        notes__ = [self.notes_dict[note] for note in notes_]
        c('notes', notes)
        c('notes_', notes_)
        c('notes__', notes__)
        return notes__
converter = Notes()


if __name__ == '__main__':
   sing()
   print("finished")