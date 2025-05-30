def create_model(
    *,
    task,
    hidden_dim,
    time_dim,
    num_layers,
    max_atoms,
    act_fn,
    dis_emb,
    num_freqs,
    ln,
    ip,
    smooth,
    cond_dim,
    pred_atom_types,
):
    from chemeleon_dng.cspnet import CSPNet, CSPNetTask

    if task.lower() == "csp":
        task = CSPNetTask.CSP
        assert pred_atom_types is False, cond_dim <= 0
        cond_dim = 0
    elif task.lower() == "dng":
        task = CSPNetTask.DNG
        assert pred_atom_types is True, cond_dim <= 0
    elif task.lower() == "guide":
        task = CSPNetTask.GUIDE
        assert pred_atom_types is True, cond_dim > 0
    elif task.lower() == "clip":
        task = CSPNetTask.CLIP
        assert pred_atom_types is False, cond_dim <= 0
    else:
        raise ValueError(f"Unknown task {task}")
    model = CSPNet(
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_layers=num_layers,
        max_atoms=max_atoms,
        act_fn=act_fn,
        dis_emb=dis_emb,
        num_freqs=num_freqs,
        ln=ln,
        ip=ip,
        smooth=smooth,
        cond_dim=cond_dim,
        pred_atom_types=pred_atom_types,
        task=task,
    )
    return model


def create_diffusion_module(
    *,
    task,
    model_configs,
    optimizer_configs,
    num_timesteps,
    # ddpm
    beta_schedule_ddpm,
    # d3pm
    beta_schedule_d3pm,
    max_atoms,
    d3pm_hybrid_coeff,
    # dsm
    sigma_begin,
    sigma_end,
):
    from chemeleon_dng.diffusion.diffusion_module import (
        DiffusionModule,
        DiffusionModuleTask,
    )
    from chemeleon_dng.diffusion.diffusion_scheduler import get_named_beta_schedule
    from chemeleon_dng.diffusion.models.d3pm import D3PM
    from chemeleon_dng.diffusion.models.ddpm import DDPM
    from chemeleon_dng.diffusion.models.dsm import DSM

    if task.lower() == "csp":
        task = DiffusionModuleTask.CSP
        model_configs["pred_atom_types"] = False
    elif task.lower() == "dng":
        task = DiffusionModuleTask.DNG
        model_configs["pred_atom_types"] = True
    elif task.lower() == "guide":
        task = DiffusionModuleTask.GUIDE
        model_configs["pred_atom_types"] = True
    else:
        raise ValueError("DiffusionModuleTask should be in `csp`, `dng`, `guide`.")
    betas_d3pm = get_named_beta_schedule(beta_schedule_d3pm, num_timesteps)
    d3pm = D3PM(
        betas=betas_d3pm, max_atoms=max_atoms, d3pm_hybrid_coeff=d3pm_hybrid_coeff
    )
    betas_ddpm = get_named_beta_schedule(beta_schedule_ddpm, num_timesteps)
    ddpm = DDPM(betas=betas_ddpm)
    dsm = DSM(num_timesteps=num_timesteps, sigma_begin=sigma_begin, sigma_end=sigma_end)
    # Set up DiffusionModule
    diffusion_module = DiffusionModule(
        task=task,
        num_timesteps=num_timesteps,
        diffusion_atom_type=d3pm if task != DiffusionModuleTask.CSP else None,
        diffusion_lattice=ddpm,
        diffusion_frac_coord=dsm,
        model_configs=model_configs,
        optimizer_configs=optimizer_configs,
    )
    return diffusion_module


def create_reinforce_module(
    *,
    diffusion_module_path,
    optimizer_configs,
    reward_type,
    rl_algorithm,
    num_group_samples,
    num_inner_batches,
    clip_ratio,
    kl_coeff,
    weight_atom_type,
    weight_lattice,
    weight_frac_coord,
    group_reward_normalize,
    reward_normalize,
):
    from chemeleon_dng.reinforce.reinforce_module import ReinforceModule

    reinforce_module = ReinforceModule(
        diffusion_module_path=diffusion_module_path,
        reward_type=reward_type,
        rl_algorithm=rl_algorithm,
        num_group_samples=num_group_samples,
        num_inner_batches=num_inner_batches,
        clip_ratio=clip_ratio,
        kl_coeff=kl_coeff,
        weight_atom_type=weight_atom_type,
        weight_lattice=weight_lattice,
        weight_frac_coord=weight_frac_coord,
        group_reward_normalize=group_reward_normalize,
        reward_normalize=reward_normalize,
        optimizer_configs=optimizer_configs,
    )
    return reinforce_module
