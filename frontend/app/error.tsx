"use client";
export default function ErrorPage({reset}: {reset: () => void}) {return <div className="state-panel" role="alert"><h1>Something went wrong</h1><p>The application could not render this analysis.</p><button className="button primary" onClick={reset}>Try again</button></div>;}
