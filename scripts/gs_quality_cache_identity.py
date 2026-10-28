"""Training windows use cohort indices; only official export names are unique."""
def validate_records(records,windows,training_windows=8337,official_windows=1860):
    expected=training_windows+official_windows
    if len(windows)!=expected or [r['window_index'] for r in records]!=list(range(expected)):
        raise ValueError('Duplicate/missing cache cohort index')
    official=[]
    for i,(row,window) in enumerate(zip(records,windows)):
        name=window['scene']+'_context'+'_'.join(map(str,window['context']))
        part='training' if i<training_windows else 'official'
        if row['scene']!=window['scene'] or row['name']!=name or row['part']!=part:
            raise ValueError('Cache/cohort identity differs')
        if part=='official':official.append(name)
    if len(set(official))!=official_windows:raise ValueError('Duplicate official export name')
    return dict(windows=expected,training_windows=training_windows,official_windows=official_windows,
        training_unique_context_names=len({r['name'] for r in records[:training_windows]}),official_unique_names=len(set(official)))
