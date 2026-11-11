use gpurs_artifact::{Target, Targets};

fn provides(target: &str, targets: &str) -> bool {
    target
        .parse::<Target>()
        .expect("valid target")
        .provides(&targets.parse().expect("valid targets"))
}

#[test]
fn membership() {
    assert!(provides("sm_89", ">=sm_80"));
    assert!(provides("sm_90a", ">=sm_80"));
    assert!(!provides("sm_75", ">=sm_80"));
    assert!(provides("sm_90a", "sm_90a | sm_100a"));
    assert!(provides("sm_100a", "sm_90a | sm_100a"));
    assert!(!provides("sm_90", "sm_90a | sm_100a"));
    assert!(!provides("sm_103a", "sm_100a"));
    assert!(provides("sm_103a", "sm_100f"));
    assert!(!provides("sm_103", "sm_100f"));
}

#[test]
fn round_trip() {
    let targets: Targets = " sm_90a |>= sm_100".parse().expect("valid targets");
    assert_eq!(targets.to_string(), "sm_90a | >=sm_100");
    assert!(">=sm_90a".parse::<Targets>().is_err());
    assert!("".parse::<Targets>().is_err());
}
