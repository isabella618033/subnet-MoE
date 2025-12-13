import bittensor

from mycelia.shared.config import WorkerConfig
from mycelia.shared.cycle import PhaseNames, PhaseResponse

subtensor = bittensor.Subtensor()


class PhaseManager:
    def __init__(self, config: WorkerConfig, subtensor: bittensor.Subtensor):
        self.config = config
        self.subtensor = subtensor
        self.phases = self.init_phases(config, PhaseNames())
        self.cycle_length = sum(p["length"] for p in self.phases)

    def init_phases(self, config, names):
        # ordered phase
        phases = [
            {"name": names.distribute, "length": config.cycle.distribute_period},
            {"name": names.train, "length": config.cycle.train_period},
            {"name": names.commit, "length": config.cycle.commit_period},
            {"name": names.submission, "length": config.cycle.submission_period},
            {"name": names.validate, "length": config.cycle.validate_period},
            {"name": names.merge, "length": config.cycle.merge_period},
        ]
        return phases

    def get_phase(self, block: int = None) -> PhaseResponse:
        if block is None:
            block = self.subtensor.block

        if block < 0:
            raise RuntimeError(f"Invalida block input block = {block}")

        cycle_index = block // self.cycle_length
        cycle_block_index = block % self.cycle_length  # 0..self.cycle_length-1

        # Walk through phases to find which one cycle_block_index is in
        current_start = 0
        for idx, phase in enumerate(self.phases):
            phase_end = current_start + phase["length"]  # exclusive
            if current_start <= cycle_block_index < phase_end:
                blocks_into_phase = cycle_block_index - current_start
                blocks_remaining = phase_end - cycle_block_index - 1

                return PhaseResponse(
                    block=block,
                    cycle_length=self.cycle_length,
                    cycle_index=cycle_index,
                    cycle_block_index=cycle_block_index,
                    phase_name=phase["name"],
                    phase_index=idx,
                    phase_start_block=current_start + cycle_index * self.cycle_length,
                    phase_end_block=phase_end - 1 + cycle_index * self.cycle_length,
                    blocks_into_phase=blocks_into_phase,
                    blocks_remaining_in_phase=blocks_remaining,
                )
            current_start = phase_end

        # Should never happen if PHASES and self.cycle_length are consistent
        raise RuntimeError("Failed to determine phase")

    def blocks_until_next_phase(self) -> dict[str, int]:
        """
        Returns a mapping:
            { phase_name: blocks_until_phase_starts_next }

        The value is how many blocks from `block` until
        the *start* of that phase (in this or the next cycle).
        """
        block = self.subtensor.block

        cycle_block_index = block % self.cycle_length  # 0..self.cycle_length-1

        result: dict[str, int] = {}

        start = 0
        for phase in self.phases:
            phase_start = start  # phase start within the cycle [0..self.cycle_length)

            if phase_start >= cycle_block_index:
                # Phase still ahead in *this* cycle
                blocks_until = phase_start - cycle_block_index
            else:
                # Phase has already passed in this cycle -> wait for next cycle
                blocks_until = (self.cycle_length - cycle_block_index) + phase_start

            result[phase["name"]] = blocks_until
            start += phase["length"]

        return result
