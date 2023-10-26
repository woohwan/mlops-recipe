#!/usr/bin/env python3

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import click
import myopslib

@click.group()
@click.version_option("1.0")
def cli():
    pass

@cli.command("retrain", help="머신러닝 모델 추가 학습")
def retrain():
    click.echo(click.style("모델 추가학습", bg="green", fg="white"))
    mae = myopslib.retrain()
    click.echo(click.style(f"추가학습 MAE: {mae}", bg="blue", fg="white"))

if __name__ == "__main__":
    cli()