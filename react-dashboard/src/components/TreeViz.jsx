import React, { useState } from 'react';
import { businessQuestion, CLASS_NAMES, CLASS_COLORS, CLASS_ICONS } from '../ml/models';



// --- Layout dimensions ---------------------------------------
const NW=190, NH=100, LW=166, LH=112, LEVEL_H=155, LEAF_MIN=200;

function countLeaves(node, d) {
  if (!node || node.isLeaf || d <= 0) return 1;
  return countLeaves(node.left, d-1) + countLeaves(node.right, d-1);
}
function buildLayout(node, x, w, depth, maxD, parent, items) {
  if (!node) return;
  const cx = x + w/2, cy = depth*LEVEL_H + 60;
  const trunc = !node.isLeaf && depth >= maxD;
  items.push({ node, cx, cy, depth, parent, leaf: node.isLeaf||trunc, trunc });
  if (!node.isLeaf && depth < maxD) {
    const ll = countLeaves(node.left, maxD-depth-1);
    const rl = countLeaves(node.right, maxD-depth-1);
    const lw = w*ll/(ll+rl);
    buildLayout(node.left,  x,   lw,   depth+1, maxD, {cx,cy}, items);
    buildLayout(node.right, x+lw, w-lw, depth+1, maxD, {cx,cy}, items);
  }
}

// --- Story path panel ----------------------------------------

// --- Main TreeViz ---------------------------------------------
export default function TreeViz({ root, displayDepth, q1=0, q2=0 }) {
  const [hovered, setHovered] = useState(null);

  if (!root) return <div style={{color:'var(--muted)',padding:20}}>No tree available.</div>;

  const classRanges = [`<$${Math.round(q1)}`, `$${Math.round(q1)}-$${Math.round(q2)}`, `>$${Math.round(q2)}`];

  const leaves = countLeaves(root, displayDepth);
  const svgW   = Math.max(leaves * LEAF_MIN, 860);
  const svgH   = (displayDepth+1)*LEVEL_H + 90;
  const items  = [];
  buildLayout(root, 0, svgW, 0, displayDepth, null, items);

  // Edge thickness proportional to traffic
  const maxSamples = root.nSamples;

  return (
    <div>
      {/* How-to-read guide */}
      <div style={{
        background:'var(--bg)', border:'1px solid var(--border)',
        borderRadius:8, padding:'10px 14px', marginBottom:10,
      }}>
        <div style={{display:'flex',gap:16,flexWrap:'wrap',alignItems:'center',marginBottom:8}}>
          <span style={{color:'var(--text2)',fontWeight:700,fontSize:10}}>How to read this tree:</span>
          <span style={{color:'var(--muted)',fontSize:9}}>Start at the <strong style={{color:'var(--c0)'}}>top node</strong> and answer each YES/NO question.</span>
          <span style={{color:'#34d399',fontWeight:600,fontSize:9}}>GREEN = YES branch</span>
          <span style={{color:'#f87171',fontWeight:600,fontSize:9}}>RED = NO branch</span>
          <span style={{color:'var(--muted)',fontSize:9}}>Thicker edge = more transactions</span>
        </div>
        <div style={{display:'flex',gap:10,flexWrap:'wrap',alignItems:'center',
          borderTop:'1px solid var(--border)',paddingTop:6}}>
          <span style={{color:'var(--muted)',fontSize:8,fontWeight:700,textTransform:'uppercase',letterSpacing:'.08em'}}>Level colours:</span>
          <div style={{display:'flex',alignItems:'center',gap:5,fontSize:9}}>
            <div style={{width:12,height:12,borderRadius:3,border:'2px solid #f59e0b',background:'#f59e0b14'}}/>
            <span style={{color:'#f59e0b',fontWeight:700}}>Amber</span>
            <span style={{color:'var(--muted)'}}>=  Quantity question (depth 0, 2, 4...)</span>
          </div>
          <div style={{display:'flex',alignItems:'center',gap:5,fontSize:9}}>
            <div style={{width:12,height:12,borderRadius:3,border:'2px solid #0d9488',background:'#0d948814'}}/>
            <span style={{color:'#0d9488',fontWeight:700}}>Teal</span>
            <span style={{color:'var(--muted)'}}>=  Unit Price question (depth 1, 3, 5...)</span>
          </div>
          <span style={{color:'var(--muted)',fontSize:8}}>|</span>
          <span style={{color:'var(--muted)',fontSize:8}}>
            Threshold values <strong style={{color:'var(--text2)'}}>INCREASE</strong> with depth  -  parent node always shows a lesser value than its children for the same feature
          </span>
        </div>
      </div>


      {/* SVG Tree */}
      <div style={{ overflowX:'auto', background:'var(--bg)', border:'1px solid var(--border)', borderRadius:10 }}>
        <svg width={svgW} height={svgH} style={{display:'block'}}>

          {/* Edges
               For Quantity/Price parents: RIGHT = YES (green), LEFT = NO (red)
               For other parents:          LEFT  = YES (green), RIGHT = NO (red)
          */}
          {items.map(({cx,cy,parent,leaf,node},i)=>{
            if (!parent) return null;
            const nh     = leaf ? LH : NH;
            const my     = (parent.cy+cy)/2;
            const isLeft = cx <= parent.cx;
            const thick  = Math.max(1.5, (node.nSamples/maxSamples)*10);

            const parentItem    = items.find(it => it.cx===parent.cx && it.cy===parent.cy);
            const parentFeat    = parentItem ? parentItem.node.feat : -1;
            const isAboveParent = parentFeat === 7 || parentFeat === 8;
            const isYesEdge     = isAboveParent ? !isLeft : isLeft;

            return (
              <path key={'e'+i}
                d={`M ${parent.cx} ${parent.cy+NH/2} C ${parent.cx} ${my} ${cx} ${my} ${cx} ${cy-nh/2}`}
                stroke={isYesEdge?'#34d39955':'#f8717155'} strokeWidth={thick} fill="none"/>
            );
          })}

          {/* YES / NO labels on every edge
               For Quantity (feat 8) and Unit Price (feat 7) nodes:
                 LEFT  = feature <= threshold = NO  ("more than X?" -> not exceeded)
                 RIGHT = feature >  threshold = YES ("more than X?" -> exceeded)
               For all other feature types:
                 LEFT = YES, RIGHT = NO (standard CART convention)
          */}
          {items.map(({cx,cy,parent,depth,leaf},i)=>{
            if (!parent||depth===0) return null;
            const isLeft = cx <= parent.cx;
            const lx = (cx*0.42+parent.cx*0.58) + (isLeft?-20:20);
            const ly = (cy*0.3+parent.cy*0.7) - 4;

            // Look up the parent node's feature to decide label direction
            const parentItem    = items.find(it => it.cx===parent.cx && it.cy===parent.cy);
            const parentFeat    = parentItem ? parentItem.node.feat : -1;
            const isAboveParent = parentFeat === 7 || parentFeat === 8;
            // Swap for "above/more-than" questions: LEFT=NO, RIGHT=YES
            const showYes = isAboveParent ? !isLeft : isLeft;

            return (
              <g key={'yl'+i}>
                <rect x={lx-16} y={ly-9} width={32} height={14} rx={5}
                  fill={showYes?'#34d39920':'#f8717120'}
                  stroke={showYes?'#34d39966':'#f8717166'} strokeWidth={1}/>
                <text x={lx} y={ly} textAnchor="middle"
                  fill={showYes?'#34d399':'#f87171'} fontSize={8} fontWeight="800">
                  {showYes?'YES':'NO'}
                </text>
              </g>
            );
          })}

          {/* Nodes */}
          {items.map(({node,cx,cy,leaf,trunc},i)=>{
            const cl     = node.majority;
            const clr    = CLASS_COLORS[cl];
            const tot    = node.classDist.reduce((s,v)=>s+v,0)||1;
            const pur    = Math.round(node.classDist[cl]/tot*100);
            const isH    = hovered===i;


            if (trunc) return (
              <g key={'n'+i}>
                <rect x={cx-LW/2} y={cy-30} width={LW} height={60} rx={8}
                  fill="var(--card2)" stroke="var(--border2)" strokeWidth={1.5} strokeDasharray="4 3"/>
                <text x={cx} y={cy-4} textAnchor="middle" fill="var(--muted)" fontSize={13}>...</text>
                <text x={cx} y={cy+11} textAnchor="middle" fill="var(--muted)" fontSize={8}>Use level buttons above to expand</text>
              </g>
            );

            if (leaf) {
              // -- Leaf (Prediction) node --
              const range = classRanges[cl];
              return (
                <g key={'n'+i} onMouseEnter={()=>setHovered(i)} onMouseLeave={()=>setHovered(null)}>
                  <rect x={cx-LW/2+3} y={cy-LH/2+3} width={LW} height={LH} rx={14} fill={clr} fillOpacity={0.07}/>
                  <rect x={cx-LW/2} y={cy-LH/2} width={LW} height={LH} rx={14}
                    fill={clr+'1a'} stroke={clr} strokeWidth={isH?2.5:2}/>

                  {/* Prediction label */}
                  <text x={cx} y={cy-32} textAnchor="middle" fill={clr} fontSize={14} fontWeight="900"
                    letterSpacing="0.6">{CLASS_ICONS[cl]} {CLASS_NAMES[cl].toUpperCase()}</text>

                  {/* Dollar range */}
                  <text x={cx} y={cy-14} textAnchor="middle" fill="white" fontSize={10} fontWeight="700">{range}</text>

                  {/* Transaction count */}
                  <text x={cx} y={cy+2} textAnchor="middle" fill="var(--text2)" fontSize={9}>
                    {node.nSamples} transactions ({Math.round(node.nSamples/root.nSamples*100)}%)
                  </text>

                  {/* Confidence bar */}
                  <rect x={cx-54} y={cy+11} width={108} height={6} rx={3} fill="var(--border2)"/>
                  <rect x={cx-54} y={cy+11} width={pur*1.08} height={6} rx={3} fill={clr} fillOpacity={0.85}/>
                  <text x={cx} y={cy+25} textAnchor="middle" fill={clr} fontSize={8} fontWeight="700">
                    {pur}% of this path are truly {CLASS_NAMES[cl]}
                  </text>

                  {/* Class mix bar */}
                  {(()=>{
                    const bw=LW-20,bx=cx-bw/2,by=cy+LH/2-10; let off=0;
                    return [0,1,2].map(c=>{
                      const p=node.classDist[c]/tot; if(!p) return null;
                      const el=<rect key={c} x={bx+off*bw} y={by} width={p*bw} height={5} fill={CLASS_COLORS[c]} rx={c===0?2:0}/>;
                      off+=p; return el;
                    });
                  })()}
                </g>
              );
            }

            // -- Decision (split) node --
            const bq   = businessQuestion(node.feat, node.thresh);
            const lPct = node.left  ? Math.round(node.left.nSamples /tot*100) : 0;
            const rPct = node.right ? Math.round(node.right.nSamples/tot*100) : 0;

            // Visual identity per feature type
            // Qty (feat 8)  = amber/gold  (#f59e0b)
            // Price (feat 7) = teal/white  (#0d9488)
            // Other features = default cyan (#38bdf8)
            const isQtyNode   = node.feat === 8;
            const isPriceNode = node.feat === 7;
            const nodeAccent  = isQtyNode ? '#f59e0b' : isPriceNode ? '#0d9488' : 'var(--c0)';
            const nodeFill    = isQtyNode ? '#f59e0b14' : isPriceNode ? '#0d948814' : 'var(--surface)';
            const nodeFillHov = isQtyNode ? '#f59e0b22' : isPriceNode ? '#0d948822' : 'var(--card2)';
            const nodeTypeLabel = isQtyNode ? 'Quantity' : isPriceNode ? 'Unit Price' : 'Other';
            const bw=NW-22, bx=cx-bw/2, by=cy+NH/2-10; let boff=0;

            return (
              <g key={'n'+i} onMouseEnter={()=>setHovered(i)} onMouseLeave={()=>setHovered(null)}>
                {/* Outer glow ring  -  distinct per feature type */}
                <rect x={cx-NW/2-4} y={cy-NH/2-4} width={NW+8} height={NH+8} rx={14}
                  fill="none" stroke={nodeAccent} strokeWidth={isH?3:2} strokeOpacity={isH?0.6:0.35}
                  strokeDasharray={isPriceNode?"0":"0"}/>

                {/* Shadow */}
                <rect x={cx-NW/2+3} y={cy-NH/2+3} width={NW} height={NH} rx={10}
                  fill={nodeAccent} fillOpacity={0.08}/>

                {/* Main box */}
                <rect x={cx-NW/2} y={cy-NH/2} width={NW} height={NH} rx={10}
                  fill={isH ? nodeFillHov : nodeFill}
                  stroke={nodeAccent}
                  strokeWidth={isH?2.5:2}/>

                {/* Feature type badge (top left) */}
                <rect x={cx-NW/2+5} y={cy-NH/2+5} rx={4}
                  width={Math.min(nodeTypeLabel.length*5.8+12, NW-12)} height={15}
                  fill={nodeAccent} fillOpacity={0.18}
                  stroke={nodeAccent} strokeWidth={0.8} strokeOpacity={0.6}/>
                <text x={cx-NW/2+9} y={cy-NH/2+15} fill={nodeAccent} fontSize={8} fontWeight="800">
                  {nodeTypeLabel}{bq.isKeyDriver?' - KEY DRIVER':''}
                </text>

                {/* Question text */}
                {bq.question.length > 34 ? (
                  <>
                    <text x={cx} y={cy-20} textAnchor="middle" fill="var(--text)" fontSize={10} fontWeight="700">
                      {bq.question.slice(0,35)}
                    </text>
                    <text x={cx} y={cy-7} textAnchor="middle" fill="var(--text)" fontSize={10} fontWeight="700">
                      {bq.question.slice(35)}
                    </text>
                  </>
                ) : (
                  <text x={cx} y={cy-12} textAnchor="middle" fill="var(--text)" fontSize={10} fontWeight="700">
                    {bq.question}
                  </text>
                )}

                {/* YES/NO split percentages */}
                <text x={cx-NW/2+8} y={cy+9} fill="#34d399" fontSize={9} fontWeight="700">YES {lPct}%</text>
                <text x={cx+NW/2-8} y={cy+9} textAnchor="end" fill="#f87171" fontSize={9} fontWeight="700">NO {rPct}%</text>

                {/* Info gain */}
                <text x={cx} y={cy+23} textAnchor="middle" fill="var(--muted)" fontSize={8}>
                  {node.nSamples} samples | gain {node.infoGain.toFixed(3)}
                </text>

                {/* Class distribution bar */}
                {[0,1,2].map(c=>{
                  const p=node.classDist[c]/tot; if(!p) return null;
                  const el=<rect key={c} x={bx+boff*bw} y={by} width={p*bw} height={5} fill={CLASS_COLORS[c]} rx={c===0?2:0}/>;
                  boff+=p; return el;
                })}
              </g>
            );
          })}
        </svg>

        {/* Legend bar */}
        <div style={{ display:'flex', gap:12, padding:'8px 14px', borderTop:'1px solid var(--border)', flexWrap:'wrap', alignItems:'center' }}>
          <div style={{display:'flex',alignItems:'center',gap:6,fontSize:9}}>
            <div style={{width:14,height:14,borderRadius:4,border:'2.5px solid #f59e0b',background:'#f59e0b14'}}/>
            <span style={{color:'#f59e0b',fontWeight:700}}>Amber box = Quantity node</span>
            <span style={{color:'var(--muted)',fontSize:8}}>(even depths 0, 2, 4 ...)</span>
          </div>
          <div style={{display:'flex',alignItems:'center',gap:6,fontSize:9}}>
            <div style={{width:14,height:14,borderRadius:4,border:'2.5px solid #0d9488',background:'#0d948814'}}/>
            <span style={{color:'#0d9488',fontWeight:700}}>Teal box = Unit Price node</span>
            <span style={{color:'var(--muted)',fontSize:8}}>(odd depths 1, 3, 5 ...)</span>
          </div>
          <span style={{color:'var(--muted)',fontSize:7.5,flex:1,textAlign:'right'}}>
            Threshold values increase with depth &bull; parent always shows a lesser value than its children on each feature
          </span>
        </div>
      </div>
    </div>
  );
}
