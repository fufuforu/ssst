#!/usr/bin/env python3
"""Context refer evaluation entry; novel cameras are passed explicitly by callers."""
import argparse
import json

from object_locus_text_refer.evaluation import evaluate_records


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--records-json",required=True,help="rows emitted by the frozen renderer adapter")
    p.add_argument("--output",required=True)
    p.add_argument("--scope",choices=["context","explicit-novel-camera"],default="context")
    args=p.parse_args()
    rows=json.load(open(args.records_json))
    result=evaluate_records(rows)
    result["scope"]=args.scope
    result["novel_protocol_status"]="not aligned to official target-view list" if args.scope=="context" else "explicit camera only; not official novel benchmark"
    with open(args.output,"w") as f: json.dump(result,f,indent=2)

if __name__=="__main__": main()
