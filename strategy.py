from ml import predict_lap_time

COMPOUNDS = {"Soft":{"pace":-0.45,"life":17,"pit_loss":22.5},"Medium":{"pace":0,"life":27,"pit_loss":22.5},"Hard":{"pace":0.35,"life":39,"pit_loss":22.5}}

def compare_strategies(payload, model_path):
    laps = int(payload.get("race_laps", 52))
    if not 5 <= laps <= 100: raise ValueError("Race distance must be between 5 and 100 laps.")
    base = {k:payload.get(k,v) for k,v in {
      "track":"Silverstone","fuel_kg":45,"air_temp_c":25,"track_temp_c":32,"rainfall":0,
      "traffic":.2,"driver_level":.7,"setup_downforce":.5}.items()}
    options = [
      [{"compound":"Soft","stint":min(15,laps)}, {"compound":"Medium","stint":max(1,laps-min(15,laps))}],
      [{"compound":"Medium","stint":min(26,laps)}, {"compound":"Hard","stint":max(1,laps-min(26,laps))}],
      [{"compound":"Hard","stint":min(35,laps)}, {"compound":"Medium","stint":max(1,laps-min(35,laps))}],
      [{"compound":"Soft","stint":max(1,laps//3)}, {"compound":"Medium","stint":max(1,laps//3)}, {"compound":"Hard","stint":max(1,laps-2*(laps//3))}],
    ]
    names = ["Aggressive: Soft → Medium","Balanced: Medium → Hard","Conservative: Hard → Medium","Three-stop-risk: Soft → Medium → Hard"]
    results=[]
    for name,stints in zip(names,options):
        total=0; pitstops=max(0,len(stints)-1); rows=[]; lap_index=0
        for s in stints:
            compound=s["compound"]; count=s["stint"]
            for age in range(count):
                inputs={**base,"compound":compound,"tyre_age":age,"fuel_kg":max(0,float(base["fuel_kg"])-lap_index*1.8)}
                pred=predict_lap_time(inputs,model_path)["predicted_lap_seconds"]
                # Approximate tyre degradation and pace offset are explicitly transparent heuristics.
                lap=pred + max(0,age-COMPOUNDS[compound]["life"]*.45)*.045
                total += lap; lap_index += 1
            rows.append({"compound":compound,"laps":count})
        total += pitstops*22.5
        results.append({"strategy":name,"stints":rows,"pit_stops":pitstops,"total_seconds":round(total,2),
                        "total_formatted":f"{int(total//60)}:{total%60:06.2f}",
                        "assumptions":"Predicted lap pace + simple tyre-age degradation + 22.5 s per pit stop; not a simulator-grade strategy model."})
    results.sort(key=lambda x:x["total_seconds"])
    for i,r in enumerate(results,1): r["rank"]=i
    return {"race_laps":laps,"strategies":results,"best":results[0],
            "warning":"Illustrative comparison. Safety cars, traffic, weather evolution, tyre rules, undercut/overcut and compound-specific real data are not fully modelled."}
