export interface TuneTask {
  name: string;
  cwd: string;
  kernelPath: string;
  buildCmd: string;
  profileCmd: string;
}

export interface TuneResult {
  round: number;
  success: boolean;
  profileOutput: string;
  errorMessage?: string;
}
