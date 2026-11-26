import csv
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from analyze_subjective_results import load_key, read_responses, summarize, exact_sign_p, write_outputs, normalize_mapping

class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        permutations=[p for p in itertools.permutations(('real','markov','neural','attention')) if p[1]=='attention' or p[2]=='attention']
        self.keyfile=self.root/'key.json'
        self.keyfile.write_text(json.dumps({'scenarios':[{'external_id':f's{i}','collection_order_v2':i,'mapping':dict(zip('ABCD',permutations[i%12]))} for i in range(240)]}))
        self.key=load_key(self.keyfile)
    def read(self,spec):
        path=self.root/'input.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['participant_id','external_id','A','B','C','D','gaming_frequency']);w.writeheader()
            for pid,sid,values,group in spec:
                w.writerow({'participant_id':pid,'external_id':sid,'gaming_frequency':group,**{k:values[v] for k,v in self.key[sid]['mapping'].items()}})
        return read_responses(path,self.key)
    def test_flow_and_sensitivity(self):
        good={'real':5,'markov':2,'neural':4,'attention':1}
        bad={**good,'attention':3,'real':1}
        incomplete={**good,'neural':''}
        rows,profiles=self.read([('p1','s0',good,'weekly'),('p1','s1',bad,'weekly'),('p2','s2',bad,'rare'),('p2','s3',incomplete,'rare'),('p3','s4',good,'weekly')])
        s=summarize(rows,profiles)
        self.assertEqual(s['flow']['attention_passed'],2)
        self.assertEqual(s['flow']['attention_failed'],2)
        self.assertEqual(s['flow']['incomplete_rows'],1)
        self.assertEqual(s['flow']['participants_with_no_passed_scenario'],1)
        self.assertEqual(s['main']['conditions']['real']['mean'],5)
        self.assertEqual(s['sensitivity']['conditions']['real']['mean'],3)
        self.assertEqual(s['main']['n_participants'],2)
        self.assertEqual(s['main']['paired_comparisons'][0]['p_exact_sign'],.5)
        self.assertEqual(s['main']['paired_comparisons'][0]['p_holm'],1)
        self.assertEqual(s['exploratory_profiles'],[])
        write_outputs(s,self.root/'out')
        self.assertEqual(len(list((self.root/'out').glob('*'))),4)
    def test_weighting_and_subgroup(self):
        good={'real':5,'markov':1,'neural':3,'attention':1}
        low={**good,'real':1}
        spec=[('p0','s0',low,'weekly'),('p0','s1',low,'weekly')]+[(f'p{i}',f's{i+1}',good,'weekly') for i in range(1,5)]
        rows,profiles=self.read(spec);s=summarize(rows,profiles)
        self.assertAlmostEqual(s['main']['conditions']['real']['mean'],22/6)
        self.assertAlmostEqual(s['main']['conditions']['real']['participant_weighted_mean'],21/5)
        self.assertEqual(len(s['exploratory_profiles']),1)
        self.assertEqual(s['main']['paired_comparisons'][0]['ties'],1)
    def test_duplicate_and_invalid(self):
        good={'real':5,'markov':2,'neural':4,'attention':1}
        with self.assertRaises(ValueError): self.read([('p1','s0',good,''),('p2','s0',good,'')])
        with self.assertRaises(ValueError): self.read([('p1','s0',{**good,'attention':6},'')])
    def test_empty_and_sign(self):
        s=summarize([],{})
        self.assertIsNone(s['flow']['attention_pass_rate'])
        self.assertIsNone(s['main']['conditions']['real']['mean'])
        self.assertEqual(exact_sign_p(5,0),.0625)
        self.assertIsNone(exact_sign_p(0,0))
        self.assertEqual(normalize_mapping({'real':'A','markov':'D','attention':'B','neural':'C'})['B'],'attention')

if __name__=='__main__': unittest.main()
