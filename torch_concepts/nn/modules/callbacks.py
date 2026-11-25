"""Lightning callbacks that schedule parts of the objective during training."""

import pytorch_lightning as pl


class LossWeightWarmup(pl.Callback):
    """Ramp one :class:`~torch_concepts.nn.CompositeLoss` weight up over epochs.

    The weight moves linearly from ``start`` to the value the loss was
    configured with, reaching it after ``epochs`` epochs. Per epoch rather than
    per batch, so the schedule does not shift with the batch size.

    The usual reason to want this is a VAE's KL term: at full strength from the
    first step it is trivially minimised by ignoring the input — the guide
    matches the prior, ``z`` carries nothing, and dimensions lost that way
    rarely come back. ``free_bits`` on
    :class:`~torch_concepts.nn.KLDivergenceLoss` addresses the same failure from
    the other end.

    Args:
        term (str or int): The term to schedule — a name from the loss's
            ``term_names`` (each term's class name unless the loss was given
            ``names``), or an index into them.
        epochs (int): Epochs taken to reach the configured weight. ``0``
            disables the schedule.
        start (float): Weight at epoch 0. Default ``0.0``.

    Example:
        >>> from torch_concepts.nn import LossWeightWarmup
        >>> callback = LossWeightWarmup(term='KLDivergenceLoss', epochs=5)
    """

    def __init__(self, term, epochs: int, start: float = 0.0):
        super().__init__()
        self.term = term
        self.epochs = int(epochs)
        self.start = float(start)
        self.index = self.target = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        loss = pl_module.loss
        if self.target is None:
            # Read the configured weight once: from the second epoch on, the
            # stored value is whatever this callback last wrote.
            self.index = (self.term if isinstance(self.term, int)
                          else loss.term_names.index(self.term))
            self.target = float(loss.weights[self.index])
        # `epochs=0` keeps the configured weight, so a sweep can include "no
        # warmup" as a point without swapping the callback out.
        fraction = min(1.0, trainer.current_epoch / self.epochs) if self.epochs else 1.0
        weight = self.start + fraction * (self.target - self.start)
        loss.weights[self.index] = weight
        pl_module.log(f"weight_{loss.term_names[self.index]}", weight)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        # Evaluate with the configured weight, so `val_loss` stays comparable
        # across epochs and a checkpoint or early stop monitoring it is not
        # biased toward the down-weighted warm-up epochs.
        if self.target is not None:
            self._scheduled = pl_module.loss.weights[self.index]
            pl_module.loss.weights[self.index] = self.target

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.target is not None:
            pl_module.loss.weights[self.index] = self._scheduled

    on_test_epoch_start = on_validation_epoch_start
    on_test_epoch_end = on_validation_epoch_end


class TemporalCMRMonitor(pl.Callback):
    """Track a temporal CMR's concepts and learned rules over training.

    Lightning decides *when* (end of every training epoch, after validation)
    and *where* (the trainer's loggers); matplotlib draws the figures.

    Each epoch it logs scalars through ``pl_module.log``, so every logger
    charts them: ``rules/robustness_scale`` and, per learned rule,
    ``rules/{task}/r{i}/start``, ``.../end`` and ``.../selection`` (mean
    selection weight on the monitored sequences).

    Every ``every_n_epochs`` epochs (and at the last epoch) it renders

    * ``plots/concept_traces``: input, predicted vs. true concepts and task
      probabilities for ``n_examples`` fixed validation sequences;
    * ``plots/concept_traces_rare``: the same for one sequence per task that
      carries the task's minority label (e.g. a window *with* an arrhythmia),
      found by scanning up to ``rare_scan_batches`` validation batches;
    * ``plots/rule_bank``: concept weights and intervals of a learned SNF bank;

    and at the end of fitting ``plots/rule_history``: interval trajectories.

    Figures go to every logger that accepts images (``WandbLogger.log_image``,
    TensorBoard ``add_image``). They are written as PNG files to ``save_dir``
    when given, or to ``<log_dir>/plots`` when no logger takes images.

    Args:
        every_n_epochs: Figure frequency; scalars are logged every epoch.
        n_examples: Number of sequences in the concept-trace figure.
        save_dir: Optional directory for PNG copies of every figure.
        log_figures: Set ``False`` to log scalars only.
        rare_examples: Also plot one minority-label sequence per task.
        rare_scan_batches: Batches scanned to find them.
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        n_examples: int = 3,
        save_dir=None,
        log_figures: bool = True,
        rare_examples: bool = True,
        rare_scan_batches: int = 50,
    ):
        super().__init__()
        if every_n_epochs < 1 or n_examples < 1:
            raise ValueError("every_n_epochs and n_examples must be positive.")
        self.every_n_epochs = every_n_epochs
        self.n_examples = n_examples
        self.save_dir = save_dir
        self.log_figures = log_figures
        self.rare_examples = rare_examples
        self.rare_scan_batches = rare_scan_batches
        self._rare = None
        self.history = {"epoch": [], "start": [], "end": [], "robustness_scale": []}
        self._examples = None

    @staticmethod
    def _decoder(pl_module):
        from .high.rules.snf import SNFRuleDecoder

        decoder = pl_module.model.rule_decoder
        return decoder if isinstance(decoder, SNFRuleDecoder) else None

    @staticmethod
    def _loader(trainer):
        loaders = trainer.val_dataloaders or trainer.train_dataloader
        return loaders[0] if isinstance(loaders, (list, tuple)) else loaders

    @staticmethod
    def select_rare(targets, task_names):
        """One sequence per task whose label is that task's minority class.

        Returns ``[(index, task, label, rate)]`` with distinct indices; tasks
        with a single class among ``targets`` are skipped.
        """
        chosen, used = [], set()
        rates = targets.float().mean(0)
        for task_index, task in enumerate(task_names):
            rate = float(rates[task_index])
            if rate in (0.0, 1.0):
                continue
            label = 1.0 if rate < 0.5 else 0.0
            matches = (targets[:, task_index] == label).nonzero().flatten().tolist()
            candidates = [index for index in matches if index not in used]
            if candidates:
                used.add(candidates[0])
                minority = rate if label == 1.0 else 1.0 - rate
                chosen.append((candidates[0], task, int(label), minority))
        return chosen

    def _rare_batch(self, trainer, pl_module):
        if self._rare is None:
            import torch

            parts = []
            for index, batch in enumerate(self._loader(trainer)):
                if index >= self.rare_scan_batches:
                    break
                parts.append(pl_module.unpack(batch))
            inputs, concepts, targets = (torch.cat(p) for p in zip(*parts))
            chosen = self.select_rare(targets, pl_module.model.task_names)
            indices = [index for index, _, _, _ in chosen]
            titles = [
                f"{task} = {label}  ({rate:.0%} of scanned)"
                for _, task, label, rate in chosen
            ]
            self._rare = (inputs[indices], concepts[indices], targets[indices], titles)
        inputs, concepts, targets, titles = self._rare
        device = pl_module.device
        return inputs.to(device), concepts.to(device), targets.to(device), titles

    def _example_batch(self, trainer, pl_module):
        if self._examples is None:
            inputs, concepts, targets = pl_module.unpack(next(iter(self._loader(trainer))))
            n = min(self.n_examples, inputs.shape[0])
            self._examples = (inputs[:n], concepts[:n], targets[:n])
        return tuple(t.to(pl_module.device) for t in self._examples)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        import numpy as np
        import torch

        epoch = trainer.current_epoch
        inputs, concepts, targets = self._example_batch(trainer, pl_module)
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            output = pl_module.model(inputs)
        pl_module.train(was_training)

        decoder = self._decoder(pl_module)
        selection = output.rule_selection.mean(0).cpu().numpy()
        if decoder is not None:
            with torch.no_grad():
                start, end = (t.cpu().numpy() for t in decoder._intervals())
                scale = float(decoder.robustness_scale)
            scalars = {"rules/robustness_scale": scale}
            for task_index, task in enumerate(decoder.task_names):
                for rule in range(decoder.n_rules):
                    prefix = f"rules/{task}/r{rule}"
                    scalars[f"{prefix}/start"] = float(start[task_index, rule])
                    scalars[f"{prefix}/end"] = float(end[task_index, rule])
                    scalars[f"{prefix}/selection"] = float(selection[task_index, rule])
            pl_module.log_dict(scalars)
            self.history["epoch"].append(epoch)
            self.history["start"].append(start)
            self.history["end"].append(end)
            self.history["robustness_scale"].append(scale)

        last = trainer.max_epochs is not None and epoch == trainer.max_epochs - 1
        if not self.log_figures or (epoch % self.every_n_epochs and not last):
            return
        from .high.rules import plotting

        figure = plotting.plot_concept_traces(
            inputs.cpu().numpy(),
            output.concepts.tensor.cpu().numpy(),
            concepts.cpu().numpy(),
            pl_module.model.annotations.labels,
            output.tasks.tensor.cpu().numpy(),
            targets.cpu().numpy(),
            pl_module.model.task_names,
            title=f"epoch {epoch}",
        )
        self._emit(trainer, "plots/concept_traces", figure, epoch)
        if self.rare_examples:
            rare_inputs, rare_concepts, rare_targets, titles = self._rare_batch(trainer, pl_module)
            if titles:
                pl_module.eval()
                with torch.no_grad():
                    rare_output = pl_module.model(rare_inputs)
                pl_module.train(was_training)
                figure = plotting.plot_concept_traces(
                    rare_inputs.cpu().numpy(),
                    rare_output.concepts.tensor.cpu().numpy(),
                    rare_concepts.cpu().numpy(),
                    pl_module.model.annotations.labels,
                    rare_output.tasks.tensor.cpu().numpy(),
                    rare_targets.cpu().numpy(),
                    pl_module.model.task_names,
                    title=f"epoch {epoch} · minority-label sequences",
                    column_titles=titles,
                )
                self._emit(trainer, "plots/concept_traces_rare", figure, epoch)
        if decoder is not None:
            with torch.no_grad():
                antecedent = decoder.antecedent_logits.softmax(-1).cpu().numpy()
                consequent = decoder.consequent_logits.softmax(-1).cpu().numpy()
            figure = plotting.plot_rule_bank(
                antecedent, consequent, start, end, decoder.config.max_horizon,
                decoder.task_names, decoder.annotations.labels,
                selection=np.asarray(selection), title=f"epoch {epoch}",
                kinds=[rule["kind"] for rule in decoder.export_rules()],
            )
            self._emit(trainer, "plots/rule_bank", figure, epoch)

    def on_fit_end(self, trainer, pl_module) -> None:
        import numpy as np

        decoder = self._decoder(pl_module)
        if not self.log_figures or decoder is None or len(self.history["epoch"]) < 2:
            return
        from .high.rules import plotting

        figure = plotting.plot_rule_history(
            self.history["epoch"],
            np.stack(self.history["start"]),
            np.stack(self.history["end"]),
            decoder.task_names,
            self.history["robustness_scale"],
        )
        # trainer.current_epoch has already advanced past the last epoch here.
        self._emit(trainer, "plots/rule_history", figure, self.history["epoch"][-1])

    def _emit(self, trainer, key: str, figure, epoch: int) -> None:
        emit_figure(trainer, key, figure, epoch, self.save_dir)


def emit_figure(trainer, key: str, figure, epoch: int, save_dir=None) -> None:
    """Send a matplotlib figure to every image-capable logger of ``trainer``.

    ``WandbLogger.log_image`` and TensorBoard ``add_image`` are supported. The
    figure is also saved as ``<save_dir>/<name>_epoch<NNN>.png``; without a
    ``save_dir`` it is saved to ``<log_dir>/plots`` only when no logger took it.
    """
    from pathlib import Path

    from .high.rules.plotting import figure_to_array

    image = figure_to_array(figure)
    handled = False
    for logger in trainer.loggers:
        if hasattr(logger, "log_image"):  # WandbLogger
            logger.log_image(key=key, images=[image], step=trainer.global_step)
            handled = True
        elif hasattr(getattr(logger, "experiment", None), "add_image"):  # TensorBoard
            logger.experiment.add_image(
                key, image, global_step=trainer.global_step, dataformats="HWC"
            )
            handled = True
    if save_dir is None and not handled:
        save_dir = Path(trainer.log_dir or trainer.default_root_dir) / "plots"
    if save_dir is not None:
        path = Path(save_dir) / f"{key.split('/')[-1]}_epoch{epoch:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
    figure.clear()


class TemporalInterpretabilityReport(pl.Callback):
    """Run :func:`interpretability_report` on the test set after ``trainer.test``.

    One report per entry of ``rule_modes`` (``"soft"``, ``"argmax"``,
    ``"sample"``; see :func:`interpretability_report`), all on the same
    trained model. Scalars go to every logger as ``interp/{mode}/...`` (see
    :func:`flatten_report`); the intervention curves are plotted as
    ``plots/intervention_curves_{mode}``. ``self.report`` maps each mode to
    its nested report.

    Args:
        blackbox: A trained :class:`SequenceBlackBox` for concept completeness.
        reference_rules: Rulebook for rule-language completeness.
        rule_modes: Which models to evaluate. Discrete modes need an SNF bank.
        save_dir: Optional directory for PNG copies of the figures.
        **options: Passed to :func:`interpretability_report` (``scorer``,
            ``n_repeats``, ``event_tolerance``, ``time_fractions``,
            ``n_samples``, ``seed``).
    """

    def __init__(self, blackbox=None, reference_rules=None, rule_modes=("soft",),
                 save_dir=None, **options):
        super().__init__()
        self.blackbox = blackbox
        self.reference_rules = reference_rules
        self.rule_modes = tuple(rule_modes)
        self.save_dir = save_dir
        self.options = options
        self.report = None

    def on_test_end(self, trainer, pl_module) -> None:
        from .high.models.cmrts_report import flatten_report, interpretability_report
        from .high.rules.plotting import plot_intervention_curves

        loaders = trainer.test_dataloaders
        loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
        if self.blackbox is not None:
            self.blackbox.to(pl_module.device)
        self.report = {}
        for mode in self.rule_modes:
            report = interpretability_report(
                pl_module.model, loader, self.blackbox, self.reference_rules,
                rule_mode=mode, **self.options,
            )
            self.report[mode] = report
            flat = flatten_report(report, prefix=f"interp/{mode}")
            for logger in trainer.loggers:
                logger.log_metrics(flat, step=trainer.global_step)
            figure = plot_intervention_curves(
                report["interventions"]["curves"],
                scorer=self.options.get("scorer", "auroc").upper(),
                title=f"concept interventions · {mode} rules",
            )
            emit_figure(trainer, f"plots/intervention_curves_{mode}", figure,
                        max(trainer.current_epoch - 1, 0), self.save_dir)
