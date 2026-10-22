async function api(url, options={}){
  const res = await fetch(url, options);
  const data = await res.json().catch(()=>({}));
  if(!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  updateClock();
  return data;
}
function setText(id, value){ const el=document.getElementById(id); if(el) el.textContent=value; }
function pct(v){ return v == null ? '--' : (Number(v)*100).toFixed(2)+'%'; }
function dec(v){ return v == null ? '--' : Number(v).toFixed(4); }
function updateClock(){ const d=new Date(); setText('lastUpdate', d.toLocaleTimeString()); }
function toast(message, error=false){
 const t=document.getElementById('toast'); if(!t) return;
 t.textContent=message; t.className='toast show '+(error?'error':'');
 setTimeout(()=>t.className='toast',2500);
}
async function health(){
 try{
   const d=await api('/api/health');
   const el=document.getElementById('apiStatus');
   if(el){el.textContent=d.model_loaded?'● API ONLINE':'● MODEL MISSING'; el.className='status-pill '+(d.model_loaded?'':'offline');}
 }catch(e){}
}
health(); updateClock();
