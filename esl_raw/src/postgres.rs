use ergnomics::*;
use eyre::Result;
use futures_util::TryStreamExt;
use quote_str::{quote_into, WritableIterExt};
use quote_str_macros::quote;
use sqlx::postgres::{PgHasArrayType, PgPool};
use sqlx::{query, query_as, Row, Type};

pub type StrategyOptimizationId = u8;
pub type StrategyId = i16;

pub struct Postgres {
    pub pool: PgPool,
}

impl Postgres {
    pub async fn connect() -> Result<Self> {
        let pool = PgPool::connect("postgresql://localhost:5432/DB1?user=leon_s&password=Letmein!")
            .await?;
        Ok(Self { pool })
    }

    pub async fn new_strategy_optimization(
        &self,
        strategy_name: &str,
        variables: &[merovingian::variable::NamedVariable],
    ) -> Result<StrategyOptimizationId> {
        let username = whoami::username();
        let variables = variables
            .iter()
            .map(|x| NamedVariable {
                name: x.name.clone(),
                min: x.min,
                max: x.max,
                step: x.step,
            })
            .collect_vec();
        let strategy_id = self.upsert_strategy(strategy_name).await?;
        let mut rows = query("select * from new_strategy_optimization($1, $2, $3)")
            .bind(variables)
            .bind(username)
            .bind(strategy_id)
            .fetch(&self.pool);
        let row = rows.try_next().await?.ok()?; // .ok()?;
        let row: i16 = row.try_get(0)?;
        Ok(row.try_into()?)
    }

    async fn upsert_strategy(&self, name: &str) -> Result<StrategyId> {
        let mut rows = query(
            r#"
WITH e AS (
    INSERT INTO strategies (name)
        VALUES ($1)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
)
SELECT *
FROM e
UNION
SELECT id
FROM strategies
WHERE name = $1"#,
        )
        .bind(name)
        .fetch(&self.pool);
        let row = rows.try_next().await?.ok()?;
        let row: i16 = row.try_get(0)?;
        Ok(row.try_into()?)
    }
}

#[derive(sqlx::Type)]
pub struct NamedVariable {
    // order is important
    pub min: f32,
    pub max: f32,
    pub step: f32,
    pub name: String,
}

impl PgHasArrayType for NamedVariable {
    fn array_type_info() -> sqlx::postgres::PgTypeInfo {
        ::sqlx::postgres::PgTypeInfo::with_name("NamedVariable[]")
    }
}
